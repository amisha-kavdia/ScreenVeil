import unittest

from screenveil.classify import (SensitiveTextClassifier, ClassifierConfig, group_lines,
                                 luhn_valid, verhoeff_check_digit, verhoeff_valid)
from screenveil.types import Box, Line, Word


def line(text, x=0, y=0, cw=10, h=20):
    """Build a Line from text, one Word per space-separated token."""
    words, cx = [], x
    for tok in text.split(" "):
        words.append(Word(tok, Box(cx, y, cw * len(tok), h)))
        cx += cw * (len(tok) + 1)
    return Line(words)


def kinds(text, **cfg):
    c = SensitiveTextClassifier(ClassifierConfig(**cfg))
    return sorted(f.kind for f in c.classify_line(line(text)))


def aadhaar():
    body = "23456789012"
    return body + verhoeff_check_digit(body)


class Checksums(unittest.TestCase):
    def test_verhoeff(self):
        self.assertTrue(verhoeff_valid("2363"))          # textbook example
        self.assertFalse(verhoeff_valid("2364"))
        self.assertTrue(verhoeff_valid(aadhaar()))

    def test_luhn(self):
        self.assertTrue(luhn_valid("4111111111111111"))
        self.assertFalse(luhn_valid("4111111111111112"))


class Rules(unittest.TestCase):
    def test_aadhaar_needs_valid_checksum(self):
        a = aadhaar()
        self.assertEqual(kinds(f"Aadhaar {a[:4]} {a[4:8]} {a[8:]}"), ["AADHAAR"])
        bad = a[:-1] + str((int(a[-1]) + 1) % 10)
        self.assertNotIn("AADHAAR", kinds(f"Order {bad[:4]} {bad[4:8]} {bad[8:]}"))

    def test_card_luhn(self):
        self.assertEqual(kinds("Card 4111 1111 1111 1111"), ["CARD"])
        self.assertEqual(kinds("Ref 4111 1111 1111 1112"), [])

    def test_pan_ifsc_upi(self):
        self.assertEqual(kinds("PAN ABCPK1234Z"), ["PAN"])
        self.assertEqual(kinds("IFSC AURB0001234"), ["IFSC"])
        self.assertEqual(kinds("pay priya.s@okaxis"), ["UPI_ID"])

    def test_context_rules(self):
        self.assertEqual(kinds("Your OTP is 482913"), ["OTP"])
        self.assertEqual(kinds("Meeting at 482913"), [])
        self.assertEqual(kinds("Account no 50230118774261"), ["BANK_ACCOUNT"])

    def test_label_in_neighbouring_line(self):
        c = SensitiveTextClassifier()
        label = line("Savings A/C No.", x=0, y=100)
        value = line("5023 0118 7742 61", x=300, y=100)
        self.assertEqual([f.kind for f in c.classify_lines([label, value])], ["BANK_ACCOUNT"])

    def test_ocr_confusions(self):
        # O for 0 and l for 1 inside a card number
        self.assertEqual(kinds("Card 4lll 1111 llll 1111"), ["CARD"])

    def test_secrets(self):
        self.assertEqual(kinds('AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"'), ["AWS_KEY"])
        self.assertIn("GITHUB_TOKEN", kinds("token ghp_" + "a1B2" * 9))
        self.assertEqual(kinds("url postgres://admin:S3cret@db.internal/prod"), ["CONN_STRING"])
        self.assertEqual(kinds("password: hunter22"), ["CREDENTIAL"])

    def test_split_token_fully_covered(self):
        # OCR split one quoted token into two words: both must be covered
        c = SensitiveTextClassifier()
        ln = line('GITHUB_TOKEN = "ghp_h8oOOL8dKLzdocJ2isAj IhKtJ0RlgLKOmxgJ"')
        f = c.classify_line(ln)
        self.assertEqual(len(f), 1)
        self.assertGreaterEqual(f[0].box.x2, ln.words[-1].box.x2 - 10)

    def test_custom_terms_and_no_false_positives(self):
        self.assertEqual(kinds("Project Falcon ships Friday", custom_terms=["project falcon"]),
                         ["CUSTOM_TERM"])
        self.assertEqual(kinds("Balance Rs 1,24,560.00 on 29 Sep 2026"), [])

    def test_preview_is_masked(self):
        c = SensitiveTextClassifier()
        f = c.classify_line(line("PAN ABCPK1234Z"))[0]
        self.assertNotIn("1234", f.preview)

    def test_partial_word_only_covers_value(self):
        c = SensitiveTextClassifier()
        ln = line("password:hunter22x")
        f = c.classify_line(ln)[0]
        self.assertGreater(f.box.x, ln.words[0].box.x)   # label stays readable


class Grouping(unittest.TestCase):
    def test_group_lines(self):
        ws = [Word("b", Box(60, 101, 30, 20)), Word("a", Box(0, 100, 50, 20)),
              Word("c", Box(0, 200, 50, 20))]
        lines = group_lines(ws)
        self.assertEqual(sorted(" ".join(w.text for w in l.words) for l in lines), ["a b", "c"])


if __name__ == "__main__":
    unittest.main()
