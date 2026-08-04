"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    # Các bài viết công khai trên Shopee Help Center.
    "https://help.shopee.vn/portal/4/article/79467",  # Bằng chứng trả hàng/hoàn tiền
    "https://help.shopee.vn/portal/4/article/79198",  # Phương thức thanh toán
    "https://help.shopee.vn/portal/4/article/79600",  # Theo dõi đơn hàng
    "https://help.shopee.vn/portal/4/article/79652",  # Đơn hàng quốc tế
    "https://help.shopee.vn/portal/4/article/79491",  # Tra cứu mã vận đơn
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

    if not result.success:
        raise RuntimeError(
            f"Crawl thất bại ({getattr(result, 'status_code', 'unknown')}): "
            f"{getattr(result, 'error_message', 'unknown error')}"
        )

    markdown = result.markdown or ""
    # Một số Help Center là SPA, crawler có thể chỉ lấy được title.
    if len(markdown.strip()) < 200:
        raise RuntimeError(
            "Nội dung crawl quá ngắn; trang có thể render bằng JavaScript "
            "hoặc URL không còn hợp lệ."
        )

    metadata = getattr(result, "metadata", {}) or {}
    title = metadata.get("title") or url.rstrip("/").split("/")[-1]
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().astimezone().isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ✗ Bỏ qua URL vì lỗi: {exc}")
            continue

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
