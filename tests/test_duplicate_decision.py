from src.nonprofit_matcher import DuplicateCheck, NonprofitRecord, decide_duplicate


class FixedEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 3
        return [
            [1.0, 0.0],
            [0.995, 0.1],
            [0.0, 1.0],
        ]


def test_reworded_donation_is_marked_as_the_existing_receipt() -> None:
    check = DuplicateCheck(
        incoming=NonprofitRecord(
            record_id="import-88",
            kind="donor_receipt",
            subject="Maya gave 75 USD to the pantry drive",
            details="Gift arrived online on August 12.",
        ),
        candidates=[
            NonprofitRecord(
                record_id="receipt-1042",
                kind="donor_receipt",
                subject="75 USD pantry donation from Maya",
                details="Online contribution received 2026-08-12.",
            ),
            NonprofitRecord(
                record_id="reminder-31",
                kind="volunteer_reminder",
                subject="Packing shift reminder",
                details="Luis starts at 9 AM Saturday.",
            ),
        ],
        threshold=0.9,
    )

    result = decide_duplicate(check, FixedEmbedder().embed)

    assert result.decision == "duplicate"
    assert result.matched_record_id == "receipt-1042"
    assert result.similarity > check.threshold
