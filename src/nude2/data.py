import PIL.Image
import csv
import hashlib
import json
import logging
import multiprocessing.dummy
import numpy as np
import os.path
import requests
import shutil
import sqlite3
import tempfile
import torch
import torchvision
import urllib.request

from collections import OrderedDict
from glob import glob
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


CACHE_DIR = os.path.expanduser(os.path.join("~", ".cache", "nude2", "data"))


IMAGES_CSV = os.path.join(CACHE_DIR, "met-images.csv")


MET_CSV = os.path.join(CACHE_DIR, "MetObjects.csv")


DB = os.path.join(CACHE_DIR, "met.db")


MET_COLS = OrderedDict([
    ("Object Number", "VARCHAR(256)"),
    ("Is Highlight", "BOOL"),
    ("Is Timeline Work", "BOOL"),
    ("Is Public Domain", "BOOL"),
    ("Object ID", "NUMBER"),
    ("Gallery Number", "NUMBER"),
    ("Department", "VARCHAR(255)"),
    ("AccessionYear", "VARCHAR(255)"),
    ("Object Name", "VARCHAR(255)"),
    ("Title", "VARCHAR(255)"),
    ("Culture", "VARCHAR(255)"),
    ("Period", "VARCHAR(255)"),
    ("Dynasty", "VARCHAR(255)"),
    ("Reign", "VARCHAR(255)"),
    ("Portfolio", "VARCHAR(255)"),
    ("Constituent ID", "VARCHAR(255)"),
    ("Artist Role", "VARCHAR(255)"),
    ("Artist Prefix", "VARCHAR(255)"),
    ("Artist Display Name", "VARCHAR(255)"),
    ("Artist Display Bio", "VARCHAR(255)"),
    ("Artist Suffix", "VARCHAR(255)"),
    ("Artist Alpha Sort", "VARCHAR(255)"),
    ("Artist Nationality", "VARCHAR(255)"),
    ("Artist Begin Date", "VARCHAR(255)"),
    ("Artist End Date", "VARCHAR(255)"),
    ("Artist Gender", "VARCHAR(255)"),
    ("Artist ULAN URL", "VARCHAR(255)"),
    ("Artist Wikidata URL", "VARCHAR(255)"),
    ("Object Date", "VARCHAR(255)"),
    ("Object Begin Date", "VARCHAR(255)"),
    ("Object End Date", "VARCHAR(255)"),
    ("Medium", "VARCHAR(255)"),
    ("Dimensions", "VARCHAR(255)"),
    ("Credit Line", "VARCHAR(255)"),
    ("Geography Type", "VARCHAR(255)"),
    ("City", "VARCHAR(255)"),
    ("State", "VARCHAR(255)"),
    ("County", "VARCHAR(255)"),
    ("Country", "VARCHAR(255)"),
    ("Region", "VARCHAR(255)"),
    ("Subregion", "VARCHAR(255)"),
    ("Locale", "VARCHAR(255)"),
    ("Locus", "VARCHAR(255)"),
    ("Excavation", "VARCHAR(255)"),
    ("River", "VARCHAR(255)"),
    ("Classification", "VARCHAR(255)"),
    ("Rights and Reproduction", "VARCHAR(255)"),
    ("Link Resource", "VARCHAR(255)"),
    ("Object Wikidata URL", "VARCHAR(255)"),
    ("Metadata Date", "VARCHAR(255)"),
    ("Repository", "VARCHAR(255)"),
    ("Tags", "VARCHAR(255)"),
    ("Tags AAT URL", "VARCHAR(255)"),
    ("Tags Wikidata URL", "VARCHAR(255)"),
])


CREATE_MET_TABLE = f"""
CREATE TABLE IF NOT EXISTS met ({", ".join(
    "  `{k}` {v}".format(k=k, v=v)
    for k, v in MET_COLS.items()
)},
    CONSTRAINT pk_met PRIMARY KEY (`Object ID`)
)"""



CREATE_MET_IMAGES_TABLE = """
CREATE TABLE IF NOT EXISTS met_images (
    `Object ID` NUMBER,
    `Image URL` VARCHAR(255),
    CONSTRAINT pk_met_images PRIMARY KEY(`Object ID`, `Image URL`)
)
"""


CREATE_MET_TAG_TABLE = """
CREATE TABLE IF NOT EXISTS met_tags (
    `Object ID` NUMBER,
    `Tag` VARCHAR(255),
    CONSTRAINT pk_met_tag PRIMARY KEY (`Object ID`, `Tag`)
)
"""

SELECT_MATCHING_TAGS_COLS = [
    "Object ID",
    "Title",
    "Object Number",
    "Object Name",
    "Is Highlight",
    "Is Timeline Work",
    "Is Public Domain",
    "Image URL",
    "Medium",
    "Tags",
    "Tag",
    "Image URL",
]


SELECT_MATCHING_TAGS = f"""
CREATE VIEW IF NOT EXISTS tagged_images AS
SELECT
    {", ".join(f"`{c}`" for c in SELECT_MATCHING_TAGS_COLS)}
FROM met
INNER JOIN met_tags USING (`Object ID`)
INNER JOIN met_images USING (`Object ID`)
"""


SELECT_TAGGED_MET_IMAGES_SQL = """
WITH matched_object_ids AS (
    SELECT DISTINCT
        `Object ID`
    FROM met_tags
    WHERE `Tag` IN ?
)
SELECT
    *
FROM met_images
INNER JOIN matched_object_ids USING (`Object ID`)
"""

def create_index_sql(table_name, cols):
    sanitized_cols = [c.lower().replace(" ", "") for c in cols]
    cols = [f"`{c}`" for c in cols]
    return f"""
    CREATE INDEX IF NOT EXISTS `idx__{table_name}__{'_'.join(sanitized_cols)}`
    ON {table_name} ({", ".join(cols)})
    """


class MetData(object):
    """Metroplitan Museum of Art Data"""

    def __init__(self, loc=DB):
        self.loc = loc
        self.conn = sqlite3.connect(self.loc)

        if not self.is_bootstrapped():
            logger.info("Bootstrapping MET data")
            self.bootstrap()

    def is_bootstrapped(self):
        """Return whether database has been bootstrapped"""
        with self.conn as conn:
            curs = conn.cursor()
            curs.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = set(row[0] for row in curs.fetchall())
            return tables == {"met", "met_images", "met_tags"}

    def bootstrap(self):
        with self.conn as conn:
            curs = conn.cursor()

            curs.execute(CREATE_MET_TABLE)
            curs.execute(create_index_sql("met", ["Object ID"]))

            with open(MET_CSV, "r") as fi:
                reader = csv.DictReader(fi)
                curs.executemany(
                    f"INSERT INTO met VALUES ({', '.join('?' * len(MET_COLS))})",
                    (tuple(row.values()) for row in reader),
                )

            curs.execute(CREATE_MET_IMAGES_TABLE)
            curs.execute(create_index_sql("met_images", ["Object ID"]))

            # XXX: executemany receives a `set` because there are dupes in the csv.
            # Just fix the csv here first
            with open(IMAGES_CSV, "r") as fi:
                reader = csv.DictReader(fi)
                curs.executemany(
                    "INSERT INTO met_images VALUES (?, ?)",
                    {tuple(row.values()) for row in reader},
                )

            curs.execute(CREATE_MET_TAG_TABLE)
            curs.execute(create_index_sql("met_tags", ["Object ID"]))
            curs.execute(create_index_sql("met_tags", ["Object ID", "Tag"]))

            curs.execute("SELECT `Object ID`, `Tags` FROM met WHERE `Tags` IS NOT NULL AND `Tags` != ''")

            rows = curs.fetchall()

            curs.executemany(
                "INSERT INTO met_tags VALUES (?, ?)",
                ((row[0], tag) for row in rows for tag in row[1].split("|")),
            )

            curs.execute(SELECT_MATCHING_TAGS)

        with self.conn as conn:
            curs = conn.cursor()
            curs.execute("VACUUM")

        logger.info("Finished bootstrapping");

    def __len__(self):
        with self.conn:
            curs = self.conn.cursor()
            curs.execute("SELECT COUNT(*) FROM met_images")
            return curs.fetchone()[0]

    def __getitem__(self, idx):
        with self.conn:
            curs = self.conn.cursor()
            curs.execute(
                "SELECT * FROM met_images LIMIT 1 OFFSET ?",
                (idx, ),
            )
            return curs.fetchone()

    def select(self, sql, params=tuple()):
        assert sql.upper().startswith("SELECT")
        with self.conn:
            curs = self.conn.cursor()
            curs.execute(sql, params)
            return curs.fetchall()

    def fetch_tag(self, tag, medium):
        """Return artworks with a given tag"""

        tag = tag or ""

        medium = medium or ""

        Q = """
        WITH filtered_images AS (
            SELECT *
            FROM tagged_images
            WHERE
                COALESCE(?, '') = ''
                OR
                `Tag` = ?
            )
            SELECT *
            FROM filtered_images
            WHERE
                `Medium` LIKE '%' || ? || '%'
        """
        with self.conn:
            curs = self.conn.cursor()
            curs.execute(Q, (tag, tag, medium, ))
            return [
                dict(zip(SELECT_MATCHING_TAGS_COLS, row))
                for row in curs.fetchall()
            ]


def retrieve_csv_data():
    """Store CSV's locally"""

    if not os.path.exists(CACHE_DIR):
        logger.info("Creating cache directory: %s", CACHE_DIR)
        os.makedirs(CACHE_DIR)

    if not os.path.exists(MET_CSV):
        logger.info("Downloading MET CSV: %s", MET_CSV)
        MET_CSV_URL = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
        urllib.request.urlretrieve(MET_CSV_URL, MET_CSV)

    if not os.path.exists(IMAGES_CSV):
        logger.info("Using MET Image CSV: %s", IMAGES_CSV)
        MET_IMAGES_CSV_URL = "https://github.com/gregsadetsky/open-access-is-great-but-where-are-the-images/raw/main/1.data/met-images.csv"
        urllib.request.urlretrieve(MET_IMAGES_CSV_URL, IMAGES_CSV)


class MetImageGetter(object):
    """Loading Cache for Met Images"""

    def __init__(self, cache_dir="~/.cache/nude2/images/", download=True):
        """Construct..."""
        self.cache_dir = os.path.expanduser(cache_dir)
        self.temp_dir = tempfile.mkdtemp(prefix="met-images-")
        self.download = download
        print(self.temp_dir)

    def __del__(self):
        shutil.rmtree(self.temp_dir)

    def fetch(self, image_url):
        """Return a PIL image """

        image_url = image_url.replace(" ", "%20")

        if not os.path.exists(self.cache_dir):
            logger.info("Creating image cache directory: %s", self.cache_dir)
            os.makedirs(self.cache_dir)

        _, suf = os.path.splitext(image_url)

        sha = hashlib.sha256()
        sha.update(image_url.encode())

        fname = f"met-image-{sha.hexdigest().lower()}{suf.lower()}"
        ftemp = os.path.join(self.temp_dir, fname)
        fpath = os.path.join(self.cache_dir, fname)

        if not os.path.exists(fpath):
            if not self.download:
                return None
            logger.debug("Downloading image: %s", image_url)
            try:
                print("Retrieving!")
                urllib.request.urlretrieve(image_url, ftemp)
                shutil.move(ftemp, fpath)
            except:
                logger.error("Failed to download image: %s", image_url)
                return None

        return fpath


class MetDataset(Dataset):
    """Configurable dataset"""

    crop = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
    ])

    tensorify = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    SELECT_QUERY = """
    SELECT
        `Object ID`,
        `Object Name`,
        `Title`,
        `Tags`,
        `Image URL`
    FROM met_images
    INNER JOIN met
    USING (`Object ID`)
    WHERE `Object Name` = 'Painting'
    """.strip()

    COUNT_QUERY = """
    SELECT
        COUNT(*)
    FROM met_images
    INNER JOIN met
    USING (`Object ID`)
    WHERE `Object Name` = 'Painting'
    """.strip()

    def __init__(self, tags, cache_dir="~/.cache/nude2/", force_refresh=False):
        self.cache_dir = os.path.expanduser(cache_dir)
        self.base_images_dir = os.path.join(self.cache_dir, "images")
        self.augmented_images_dir = os.path.join(self.cache_dir, "images-augmented")

        self.db = MetData(DB)
        self.fetcher = MetImageGetter(self.base_images_dir, download=False)

        base_rows = self.db.select(self.SELECT_QUERY)
        self.rows = list(self.augmentations(base_rows))

    five_crop = torchvision.transforms.Compose([
        torchvision.transforms.Resize(336),
        torchvision.transforms.FiveCrop(224),
    ])

    tensorify = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def fetch_crops(self, row):
        image_path = self.fetcher.fetch(row[-1])
        prefix, suffix = os.path.splitext(os.path.basename(image_path))
        crop_fnames = glob(os.path.join(self.augmented_images_dir, f"{prefix}-[0-4]-0.jpg"))

        crop_fnames = [
            os.path.join(self.augmented_images_dir, f"{prefix}-{i}-0.jpg")
            for i in range(5)
        ]

        crops = []

        if all(os.path.exists(fname) for fname in crop_fnames):
            for fname in crop_fnames:
                with PIL.Image.open(fname) as i:
                    crops.append(i)

        else:
            logger.warning("Generating new images for {prefix}")
            try:
                with PIL.Image.open(image_path) as i:
                    img = i.convert("RGB").copy()
                    crops = self.five_crop(img)
                    for i, c in enumerate(crops):
                        fpath = os.path.join(self.augmented_images_dir, f"{prefix}-{i}-0.jpg")
                        c.save(fpath)
            except PIL.Image.DecompressionBombError:
                logger.error("Decompression bomb error on image: %s", image_path)

        return crops

    def augmentations(self, rows):

        if not os.path.exists(self.augmented_images_dir):
            os.makedirs(self.augmented_images_dir)

        for row in rows:

            image_path = self.fetcher.fetch(row[-1])

            if image_path is None:
                continue

            crops = self.fetch_crops(row)

            object_id = row[0]
            original_path = image_path

            fname, suffix = os.path.splitext(os.path.basename(image_path))

            for i, c in enumerate(crops):
                aug_fname = f"{fname}-{i}-0{suffix}"
                aug_path = os.path.join(self.augmented_images_dir, aug_fname)

                if not os.path.exists(aug_path):
                    print("SAVE!")
                    c.save(aug_path)

                res = {
                    "fname": aug_fname,
                    "path": aug_path,
                    "nude": "Nude" in row[-2],
                    "base_image_path": row[-1],
                }

                yield res

    def save_augmentation_labels(self):
        augmented_labels_dir = os.path.join(self.cache_dir, "labels-augmented")

        if not os.path.exists(augmented_labels_dir):
            os.makedirs(augmented_labels_dir)

        for row in self.rows:
            image_prefix, _ = os.path.splitext(row["fname"])
            label_path = os.path.join(augmented_labels_dir, f"{image_prefix}.json")
            row["label_path"] = label_path

            with open(label_path, "w") as fo:
                json.dump(row, fo)


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        with PIL.Image.open(self.rows[idx]["path"]) as i:
            row = self.rows[idx]
            row["image"] = self.tensorify(np.array(i))
            return row


def main(concurrency, limit):
    data = MetDataset(["Nude Males", "Nude Females"])
    batches = DataLoader(data, batch_size=2)
    data.save_augmentation_labels()




def main2(concurrency, limit):
    retrieve_csv_data()
    conn = MetData(DB)
    getter = MetImageGetter(download=False)

    # Get all paintins
    rows = conn.fetch_tag("", "")

    image_urls = {
        row["Image URL"]
        for row in rows
    }

    rows = conn.select("""
    SELECT
        `Object Name`,
        COUNT(*) as cnt
    FROM tagged_images
    GROUP BY `Object Name`
    HAVING cnt > 1000
    ORDER BY cnt DESC
    LIMIT 10
    """.strip())

    for row in rows:
        print(row)

    rows = conn.select("""SELECT COUNT(*) FROM tagged_images WHERE `Object Name` = 'Painting'""".strip())

    print(rows)

    return

    with multiprocessing.dummy.Pool(concurrency) as pool:
        results = pool.map(
            getter.fetch,
            (url.replace(" ", "%20") for url in image_urls),
        )

    succeeded = len([row for row in results if row is not None])

    failed = len([row for row in results if row is None])

    print("succeeded .....", succeeded)
    print("failed ........", failed)
    print("total .........", len(results))
    print("rows ..........", len(rows))

    # styles = conn.select("""SELECT
    #     `Object Name`,
    #     COUNT(*) as cnt
    # FROM met
    # GROUP BY `Object Name`
    # HAVING cnt > 1000
    # ORDER BY cnt, `Object Name`
    # """)

    # for sty in styles:
    #     print(sty)


class MetCenterCroppedDataset(Dataset):

    image_size = 64

    tensorify = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    pilify = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
        torchvision.transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
        torchvision.transforms.ToPILImage(),
    ])

    def __init__(self, cache_dir="~/.cache/nude2/"):
        self.cache_dir = os.path.expanduser(cache_dir)
        self.image_dir = self.cache_dir
        self.fnames = glob(os.path.join(self.image_dir, "*.jpg"))
        self.cache = {}

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if idx not in self.cache:
            try:
                with PIL.Image.open(self.fnames[idx]) as i:
                    self.cache[idx] = self.tensorify(i.convert("RGB"))
            except PIL.Image.DecompressionBombError:
                print("fname =", self.fnames[idx])
                return None
            except PIL.Image.DecompressionBombWarning:
                print("!!")
        return self.cache[idx]
