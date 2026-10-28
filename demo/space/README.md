---
title: PageProof
emoji: 📖
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: How OCR damage changes answers about scanned books
---

# PageProof demo

Pick a question about one of six public-domain books and an OCR damage level. You see what the system answered when it searched the damaged text, what it answered when handed the right page, the judge's verdict on each, and the page image with the passage highlighted.

All answers are saved results from the project's evaluation runs (150 held-out test questions, Tesseract OCR at six damage levels). The page is static: it runs no model and makes no API calls.

Code, method and results: see the PageProof repository. The book texts are public domain (Project Gutenberg); everything else here is MIT-licensed.
