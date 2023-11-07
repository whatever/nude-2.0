import nude2.data
from datetime import datetime, timedelta
import time


def main():

    db = nude2.data.MetData(nude2.data.DB)

    tags = ["Sphinx", "Crucifixion", "Male Nudes", "Female Nudes", "asdasd", ""]

    mediums = ["oil", ""]

    cases = sorted(
        (tag, medium)
        for tag in tags
        for medium in mediums
    )

    for tag, medium in cases:
        start = datetime.now()
        results = db.fetch_tag(tag, medium)
        duration = datetime.now() - start
        pad = 30 - len(tag) - len(medium)
        print(f"tag={tag}, medium={medium} {'.'*pad} {duration}, {len(results)}")
