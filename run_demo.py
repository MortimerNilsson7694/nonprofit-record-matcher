import json

from src.nonprofit_matcher import DuplicateCheck, InfraiEmbedder, NonprofitRecord, decide_duplicate


def main() -> None:
    check = DuplicateCheck(
        incoming=NonprofitRecord(
            record_id="receipt-import-88",
            kind="donor_receipt",
            subject="Maya Chen donated 75 USD to the pantry drive",
            details="Online gift received on 2026-08-12; receipt requested by email.",
        ),
        candidates=[
            NonprofitRecord(
                record_id="receipt-1042",
                kind="donor_receipt",
                subject="75 USD pantry campaign contribution from Maya Chen",
                details="Web donation dated August 12, 2026. Send the donor an email receipt.",
            ),
            NonprofitRecord(
                record_id="volunteer-31",
                kind="volunteer_reminder",
                subject="Saturday packing shift for Luis",
                details="Reminder for the 9:00 AM warehouse shift on August 16.",
            ),
        ],
    )
    result = decide_duplicate(check, InfraiEmbedder().embed)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
