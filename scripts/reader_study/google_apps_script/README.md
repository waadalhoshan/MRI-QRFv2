# Google Apps Script reader interface

Files:
- `Code.gs` — server-side Apps Script
- `index.html` — web interface

Before deployment:
1. Create a native Google Sheet bound to the Apps Script project.
2. Keep sheets:
   - `Ratings`
   - `Consent_Log`
3. Set the Drive folder ID containing only `IMG_001.png` ... `IMG_160.png`.
4. Run `buildManifestFromDrive()`.
5. Confirm exactly 160 images and the complete sequential ID range.
6. Deploy as a web app, executing as the study owner.
7. Give each reader a unique rater code (e.g., RAD01/RAD02).
8. Share only the deployed app URL and rater code.

Do not expose the answer key in Drive folders accessible to readers or in the Apps Script project.

The backend upserts by rater+image ID, so revisiting an image updates rather than duplicates the rating. Progress can be reconstructed across devices when the reader uses the same code.
