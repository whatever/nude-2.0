import csv
import logging
import os.path
import requests
import sqlite3
import urllib.request

from collections import OrderedDict


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
    )})"""



CREATE_MET_IMAGES_TABLE = """
CREATE TABLE IF NOT EXISTS met_images (
    `Object ID` NUMBER,
    `Image URL` VARCHAR(255)
)
"""


CREATE_MET_TAG_TABLE = """
CREATE TABLE IF NOT EXISTS met_tags (
    `Object ID` NUMBER,
    `Tag` VARCHAR(255)
)
"""

SELECT_MATCHING_TAGS_COLS = [
    "Object ID",
    "Object Number",
    "Is Highlight",
    "Is Timeline Work",
    "Is Public Domain",
    "Image URL",
    "Medium",
    "Tags",
    "Tag",
    "Image URL",
]


SELECT_MATCHING_TAGS = """
CREATE VIEW IF NOT EXISTS tagged_images AS
SELECT
    `Object ID`,
    `Object Number`,
    `Is Highlight`,
    `Is Timeline Work`,
    `Is Public Domain`,
    `Image URL`,
    `Medium`,
    `Tags`,
    `Tag`,
    `Image URL`
FROM met
INNER JOIN met_tags USING (`Object ID`)
INNER JOIN met_images USING (`Object ID`)
"""



class MetData(object):
    """Metroplitan Museum of Art Data"""

    def __init__(self, loc=DB):
        self.loc = loc
        self.conn = sqlite3.connect(self.loc)

        if not self.is_bootstrapped():
            logging.warning("Bootstrapping MET data")
            bootstrap(self)

    def is_bootstrapped(self):
        """Return whether database has been bootstrapped"""
        with self.conn as conn:
            curs = conn.cursor()
            curs.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = set(row[0] for row in curs.fetchall())
            return tables == {"met", "met_images", "met_tags"}

    def bootstrap(self):
        curs = conn.cursor()

        curs.execute(CREATE_MET_TABLE)

        with open(MET_CSV, "r") as fi:
            reader = csv.DictReader(fi)
            curs.executemany(
                f"INSERT INTO met VALUES ({', '.join('?' * len(MET_COLS))})",
                (tuple(row.values()) for row in reader),
            )

        curs.execute(CREATE_MET_IMAGES_TABLE)

        with open(IMAGES_CSV, "r") as fi:
            reader = csv.DictReader(fi)
            curs.executemany(
                "INSERT INTO met_images VALUES (?, ?)",
                (tuple(row.values()) for row in reader),
            )

        curs.execute(CREATE_MET_TAG_TABLE)

        curs.execute("SELECT `Object ID`, `Tags` FROM met WHERE `Tags` IS NOT NULL AND `Tags` != ''")

        rows = curs.fetchall()

        curs.executemany(
            "INSERT INTO met_tags VALUES (?, ?)",
            ((row[0], tag) for row in rows for tag in row[1].split("|")),
        )

        curs.execute(SELECT_MATCHING_TAGS)

        return conn

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

    def select(self, sql):
        assert sql.upper().startswith("SELECT")
        with self.conn:
            curs = self.conn.cursor()
            curs.execute(sql)
            return curs.fetchall()

    def fetch_tag(self, tag, medium):
        with self.conn:
            curs = self.conn.cursor()
            curs.execute(
                "SELECT * FROM tagged_images WHERE `Tag` = ? LIMIT ?",
                 (tag, 1000, ),
             )
            return [
                dict(zip(SELECT_MATCHING_TAGS_COLS, row))
                for row in curs.fetchall()
                if medium in row[-4]
            ]


def retrieve_csv_data():
    """Store CSV's locally"""

    if not os.path.exists(CACHE_DIR):
        logging.warning("Creating cache directory: %s", CACHE_DIR)
        os.makedirs(CACHE_DIR)

    if not os.path.exists(MET_CSV):
        logging.warning("Downloading MET CSV: %s", MET_CSV)
        MET_CSV_URL = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
        urllib.request.urlretrieve(MET_CSV_URL, MET_CSV)

    if not os.path.exists(IMAGES_CSV):
        logging.warning("Using MET Image CSV: %s", IMAGES_CSV)
        MET_IMAGES_CSV_URL = "https://github.com/gregsadetsky/open-access-is-great-but-where-are-the-images/raw/main/1.data/met-images.csv"
        urllib.request.urlretrieve(MET_IMAGES_CSV_URL, IMAGES_CSV)


def main():

    retrieve_csv_data()

    conn = MetData(DB)

    print(type(conn))
    print(len(conn))
    print(conn[1000])
    for row in conn.fetch_tag("Sphinx", "oil"):
        print(row)
