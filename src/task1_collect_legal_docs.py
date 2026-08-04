"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path
import json
import re
from html.parser import HTMLParser

import requests
from fpdf import FPDF

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

DOCUMENTS = [
    {
        "url": "https://help.shopee.vn/portal/4/article/77251",
        "filename": "returns-refund-policy-shopee.pdf",
        "customer_role": "buyer",
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/79198",
        "filename": "payment-methods-shopee.pdf",
        "customer_role": "buyer",
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/77244",
        "filename": "privacy-policy-shopee.pdf",
        "customer_role": "both",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


class TextExtractor(HTMLParser):
    """Trích xuất text cơ bản từ trang Help Center HTML."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self):
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_pdf(html: bytes, output_path):
    parser = TextExtractor()
    parser.feed(html.decode("utf-8", errors="ignore"))
    content = parser.text()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # DejaVu supports Vietnamese Unicode characters.
    pdf.add_font("DejaVu", fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(0, 5, content or "Không trích xuất được nội dung trang.")
    pdf.output(str(output_path))


def download_document(document: dict) -> None:
    """Tải bài viết chính sách và lưu thành PDF theo yêu cầu Task 1."""
    response = requests.get(
        document["url"],
        headers={"User-Agent": "Mozilla/5.0 (RAG lab data collector)"},
        timeout=30,
    )
    response.raise_for_status()

    filepath = DATA_DIR / document["filename"]
    html_to_pdf(response.content, filepath)

    metadata_path = filepath.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_url": document["url"],
                "customer_role": document["customer_role"],
                "content_type": response.headers.get("content-type", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✓ Đã tải: {filepath}")


if __name__ == "__main__":
    setup_directory()
    for document in DOCUMENTS:
        try:
            download_document(document)
        except requests.RequestException as exc:
            print(f"✗ Không thể tải {document['url']}: {exc}")
