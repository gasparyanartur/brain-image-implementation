import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Hi")

parser = ArgumentParser()
parser.add_argument("--hi", action="store_true")
args = parser.parse_args()

if args.hi:
    logging.info("Hi")