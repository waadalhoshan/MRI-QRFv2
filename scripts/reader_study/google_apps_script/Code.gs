/**
 * MRI Readability Rating Tool — Backend
 *
 * This Apps Script should be bound to the Google Sheet that stores responses.
 *
 * Drive folder requirement:
 * - The folder below must contain ONLY the 160 blinded study images:
 *   IMG_001.png ... IMG_160.png
 * - Do NOT place KEY_DO_NOT_SHARE inside that folder.
 */

// ===================== CONFIG =====================

const IMAGES_FOLDER_ID = '1TRF5SnC9DJFATMZlD9CPADdoJ6bi_Cf7';
const EXPECTED_IMAGE_COUNT = 160;

// ==================================================


function doGet() {
  return HtmlService.createHtmlOutputFromFile('index')
    .setTitle('MRI Readability Assessment')
    .addMetaTag(
      'viewport',
      'width=device-width, initial-scale=1, viewport-fit=cover'
    );
}


/**
 * Return an existing sheet, or create it with the supplied headers.
 */
function getOrCreateSheet_(name, headers) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = ss.getSheetByName(name);

  if (!sheet) {
    sheet = ss.insertSheet(name);
    sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  }

  return sheet;
}


/**
 * RUN THIS ONCE MANUALLY from Apps Script after uploading the 160 images.
 *
 * Expected folder:
 *   IMG_001.png
 *   IMG_002.png
 *   ...
 *   IMG_160.png
 *
 * The Python preparation script has already shuffled and blinded the images.
 * This function therefore DOES NOT rename or reshuffle them.
 *
 * It creates/replaces a safe "Manifest" sheet containing only:
 *   image_id | drive_file_id | order
 *
 * No dataset/class/distortion/severity information is stored here.
 */
function buildManifestFromDrive() {
  const folder = DriveApp.getFolderById(IMAGES_FOLDER_ID);
  const files = folder.getFiles();

  const records = [];

  while (files.hasNext()) {
    const file = files.next();
    const name = file.getName();

    const match = name.match(/^(IMG_(\d{3}))\.(png|jpg|jpeg)$/i);

    if (!match) {
      continue;
    }

    records.push({
      imageId: match[1].toUpperCase(),
      order: Number(match[2]),
      driveFileId: file.getId()
    });
  }

  records.sort((a, b) => a.order - b.order);

  if (records.length !== EXPECTED_IMAGE_COUNT) {
    throw new Error(
      'Expected ' + EXPECTED_IMAGE_COUNT +
      ' blinded images, but found ' + records.length +
      '. Check the images_for_raters folder.'
    );
  }

  // Confirm the exact sequence IMG_001 ... IMG_160.
  for (let i = 0; i < EXPECTED_IMAGE_COUNT; i++) {
    const expected =
      'IMG_' + String(i + 1).padStart(3, '0');

    if (records[i].imageId !== expected) {
      throw new Error(
        'Image sequence problem. Expected ' +
        expected +
        ' but found ' +
        records[i].imageId +
        '.'
      );
    }
  }

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = ss.getSheetByName('Manifest');

  if (!sheet) {
    sheet = ss.insertSheet('Manifest');
  } else {
    sheet.clearContents();
  }

  const values = [
    ['image_id', 'drive_file_id', 'order'],
    ...records.map(r => [
      r.imageId,
      r.driveFileId,
      r.order
    ])
  ];

  sheet
    .getRange(1, 1, values.length, values[0].length)
    .setValues(values);

  Logger.log(
    'Manifest built successfully: ' +
    records.length +
    ' images.'
  );

  return records.length;
}


/**
 * Sent to the browser when the web app loads.
 *
 * Returns ONLY:
 * - blinded image ID
 * - Drive file ID
 *
 * No private condition information is sent.
 */
function getManifestForClient() {
  const sheet =
    SpreadsheetApp
      .getActiveSpreadsheet()
      .getSheetByName('Manifest');

  if (!sheet) {
    throw new Error(
      'Manifest sheet does not exist. ' +
      'Run buildManifestFromDrive() first.'
    );
  }

  const data = sheet.getDataRange().getValues();

  if (data.length <= 1) {
    return [];
  }

  return data
    .slice(1)
    .filter(row => row[0] && row[1])
    .sort((a, b) => Number(a[2]) - Number(b[2]))
    .map(row => ({
      id: String(row[0]).trim().toUpperCase(),
      driveFileId: String(row[1]).trim()
    }));
}


/**
 * Return one image as a base64 data URL for display in the browser.
 */
function getImageBase64(driveFileId) {
  const file = DriveApp.getFileById(driveFileId);
  const blob = file.getBlob();

  const base64 =
    Utilities.base64Encode(blob.getBytes());

  const mimeType =
    blob.getContentType();

  return (
    'data:' +
    mimeType +
    ';base64,' +
    base64
  );
}


/**
 * Save or UPDATE one rating.
 *
 * This prevents duplicates if a rater goes back and changes an answer.
 *
 * record = {
 *   rater,
 *   imageId,
 *   rating,
 *   orderShown
 * }
 */
function saveRating(record) {
  const allowedRatings = [
    'Readable',
    'Borderline',
    'Unreadable'
  ];

  if (
    !record ||
    !record.rater ||
    !record.imageId
  ) {
    throw new Error(
      'Incomplete rating record.'
    );
  }

  if (
    !allowedRatings.includes(record.rating)
  ) {
    throw new Error(
      'Invalid rating value.'
    );
  }

  const lock =
    LockService.getScriptLock();

  lock.waitLock(30000);

  try {
    const sheet =
      getOrCreateSheet_(
        'Ratings',
        [
          'timestamp',
          'rater',
          'image_id',
          'rating',
          'order_shown',
          'consent_signed'
        ]
      );

    const lastRow =
      sheet.getLastRow();

    const normalizedRater =
      String(record.rater)
        .trim()
        .toLowerCase();

    const normalizedImage =
      String(record.imageId)
        .trim()
        .toUpperCase();

    if (lastRow >= 2) {
      const data =
        sheet
          .getRange(
            2,
            1,
            lastRow - 1,
            6
          )
          .getValues();

      for (let i = 0; i < data.length; i++) {
        const existingRater =
          String(data[i][1])
            .trim()
            .toLowerCase();

        const existingImage =
          String(data[i][2])
            .trim()
            .toUpperCase();

        if (
          existingRater === normalizedRater &&
          existingImage === normalizedImage
        ) {
          const rowNumber = i + 2;

          sheet
            .getRange(
              rowNumber,
              1,
              1,
              6
            )
            .setValues([[
              new Date(),
              record.rater,
              normalizedImage,
              record.rating,
              Number(record.orderShown),
              true
            ]]);

          return {
            ok: true,
            action: 'updated'
          };
        }
      }
    }

    sheet.appendRow([
      new Date(),
      record.rater,
      normalizedImage,
      record.rating,
      Number(record.orderShown),
      true
    ]);

    return {
      ok: true,
      action: 'inserted'
    };

  } finally {
    lock.releaseLock();
  }
}


/**
 * Recover saved progress from the Ratings sheet.
 *
 * This means a radiologist can:
 * - close the browser,
 * - restart the computer,
 * - use another browser,
 * - or use another device,
 *
 * then enter the SAME rater code and continue.
 */
function getProgressForRater(raterName) {
  const name =
    String(raterName || '').trim();

  if (!name) {
    throw new Error(
      'Rater name/initials are required.'
    );
  }

  const ss =
    SpreadsheetApp.getActiveSpreadsheet();

  const manifestSheet =
    ss.getSheetByName('Manifest');

  if (!manifestSheet) {
    throw new Error(
      'Manifest sheet does not exist.'
    );
  }

  const manifestData =
    manifestSheet
      .getDataRange()
      .getValues();

  const manifest =
    manifestData
      .slice(1)
      .filter(row => row[0] && row[1])
      .sort(
        (a, b) =>
          Number(a[2]) - Number(b[2])
      );

  const ratingsSheet =
    ss.getSheetByName('Ratings');

  if (
    !ratingsSheet ||
    ratingsSheet.getLastRow() < 2
  ) {
    return {
      completedCount: 0,
      nextIndex: 0,
      ratings: {}
    };
  }

  const ratingData =
    ratingsSheet
      .getRange(
        2,
        1,
        ratingsSheet.getLastRow() - 1,
        6
      )
      .getValues();

  const ratings = {};

  const normalizedName =
    name.toLowerCase();

  ratingData.forEach(row => {
    const existingRater =
      String(row[1])
        .trim()
        .toLowerCase();

    if (
      existingRater === normalizedName
    ) {
      const imageId =
        String(row[2])
          .trim()
          .toUpperCase();

      ratings[imageId] = {
        rater: row[1],
        imageId: imageId,
        rating: row[3],
        orderShown: row[4]
      };
    }
  });

  let nextIndex =
    manifest.length;

  for (
    let i = 0;
    i < manifest.length;
    i++
  ) {
    const imageId =
      String(manifest[i][0])
        .trim()
        .toUpperCase();

    if (!ratings[imageId]) {
      nextIndex = i;
      break;
    }
  }

  return {
    completedCount:
      Object.keys(ratings).length,
    nextIndex: nextIndex,
    ratings: ratings
  };
}


/**
 * Log consent once per rater identifier.
 */
function logConsent(raterName) {
  const name =
    String(raterName || '').trim();

  if (!name) {
    throw new Error(
      'Rater name/initials are required.'
    );
  }

  const lock =
    LockService.getScriptLock();

  lock.waitLock(30000);

  try {
    const sheet =
      getOrCreateSheet_(
        'Consent_Log',
        [
          'timestamp',
          'rater'
        ]
      );

    const lastRow =
      sheet.getLastRow();

    if (lastRow >= 2) {
      const existing =
        sheet
          .getRange(
            2,
            2,
            lastRow - 1,
            1
          )
          .getValues()
          .flat()
          .map(
            v =>
              String(v)
                .trim()
                .toLowerCase()
          );

      if (
        existing.includes(
          name.toLowerCase()
        )
      ) {
        return {
          ok: true,
          alreadyLogged: true
        };
      }
    }

    sheet.appendRow([
      new Date(),
      name
    ]);

    return {
      ok: true,
      alreadyLogged: false
    };

  } finally {
    lock.releaseLock();
  }
}
