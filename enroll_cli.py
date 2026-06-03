"""
enroll_cli.py — Command-line face enrollment tool (headless, for SSH sessions).

Useful on Raspberry Pi when running over SSH without a display.
Captures MIN_ENCODINGS_PER_PERSON frames from the camera, averages them,
and stores the result in the same database used by the GUI.

Usage:
    python enroll_cli.py --name "Alice Smith"
    python enroll_cli.py --name "Bob Jones" --camera 1 --samples 5
    python enroll_cli.py --list
    python enroll_cli.py --remove "Alice Smith"
"""

import argparse
import logging
import sys
import time

import cv2

from config import CAMERA_INDEX, DATABASE_FILE, MIN_ENCODINGS_PER_PERSON
from database import FaceDatabase
from enroller import EnrollmentSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def enroll(name: str, camera_idx: int, samples: int) -> int:
    """Return 0 on success, 1 on failure."""
    db = FaceDatabase(DATABASE_FILE)
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error("Cannot open camera index %d.", camera_idx)
        return 1

    print(f"\n📸  Enrolling: {name}")
    print(f"    Samples required: {samples}")
    print("    Look directly at the camera…\n")

    session = EnrollmentSession(name, db, samples_required=samples)
    last_msg = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read camera frame.")
                return 1

            frame = cv2.flip(frame, 1)
            result = session.feed_frame(frame)

            if result.message != last_msg:
                print(f"  → {result.message}")
                last_msg = result.message

            if result.done:
                print()
                if result.success:
                    print(f"  ✅  {result.message}")
                    return 0
                else:
                    print(f"  ❌  {result.message}")
                    return 1

            time.sleep(0.05)
    finally:
        cap.release()


def list_people() -> None:
    db = FaceDatabase(DATABASE_FILE)
    people = db.list_people()
    if not people:
        print("No enrolled profiles found.")
        return
    print(f"\nEnrolled profiles ({len(people)}):")
    for name in people:
        print(f"  • {name}")
    print(f"\nTotal encodings: {db.encoding_count()}")


def remove_person(name: str) -> int:
    db = FaceDatabase(DATABASE_FILE)
    if db.remove_person(name):
        print(f"✅  Profile '{name}' removed.")
        return 0
    else:
        print(f"❌  Profile '{name}' not found.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Smart Door Access System — CLI enrollment tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", metavar="NAME", help="Name of the person to enroll")
    group.add_argument("--list", action="store_true", help="List enrolled profiles")
    group.add_argument("--remove", metavar="NAME", help="Remove a person's profile")

    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})"
    )
    parser.add_argument(
        "--samples", type=int, default=MIN_ENCODINGS_PER_PERSON,
        help=f"Number of face samples to capture (default: {MIN_ENCODINGS_PER_PERSON})"
    )

    args = parser.parse_args()

    if args.list:
        list_people()
        sys.exit(0)
    elif args.remove:
        sys.exit(remove_person(args.remove))
    else:
        sys.exit(enroll(args.name, args.camera, args.samples))


if __name__ == "__main__":
    main()
