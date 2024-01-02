#!/bin/bash
# Download Cityscapes data.
# Cf.
# https://www.cityscapes-dataset.com/downloads/

DATASET_DIR="cityscapes"

# Register with Cityscapes and fill in `../.env` with your own credentials.
source ../.env  # Defines `CITYSCAPES_USERNAME` and `CITYSCAPES_PASSWORD`.

echo -e "\nUsing Cityscapes credentials"
echo "username: ${CITYSCAPES_USERNAME}"
echo -e "password: ${CITYSCAPES_PASSWORD}"

echo -e "\nLogging into Cityscapes."
wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=${CITYSCAPES_USERNAME}&password=${CITYSCAPES_PASSWORD}&submit=Login" https://www.cityscapes-dataset.com/login/

echo -e '\nDownloading `gtFine_trainvaltest.zip` (~241 MB).'
wget -P $DATASET_DIR --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

echo -e '\nDownloading `leftImg8bit_trainvaltest.zip` (~11 GB).'
wget -P $DATASET_DIR --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

echo -e "\nRemoving HTTP artifacts."
rm -f cookies.txt index.html*
rm -f ${DATASET_DIR}/index.html*

echo -e '\nConfirming hash of `gtFine_trainvaltest.zip`.'
MD5_SUM=$(md5sum ${DATASET_DIR}/gtFine_trainvaltest.zip | awk '{print $1}')
MD5_SUM_REF="4237c19de34c8a376e9ba46b495d6f66"
if [ "$MD5_SUM" != "$MD5_SUM_REF" ]; then
 echo 'Hash of `gtFine_trainvaltest.zip` is incorrect! Exiting without decompressing.'
 exit 1
fi
echo 'Hash of `gtFine_trainvaltest.zip` is correct.'

echo -e '\nConfirming hash of `leftImg8bit_trainvaltest.zip`.'
MD5_SUM=$(md5sum ${DATASET_DIR}/leftImg8bit_trainvaltest.zip | awk '{print $1}')
MD5_SUM_REF="0a6e97e94b616a514066c9e2adb0c97f"
if [ "$MD5_SUM" != "$MD5_SUM_REF" ]; then
 echo 'Hash of `leftImg8bit_trainvaltest.zip` is incorrect! Exiting without decompressing.'
 exit 1
fi
echo 'Hash of `leftImg8bit_trainvaltest.zip` is correct.'

echo -e '\nDecompressing `gtFine_trainvaltest.zip`.'
unzip -n ${DATASET_DIR}/gtFine_trainvaltest.zip -d ${DATASET_DIR}/

echo -e '\nDecompressing `leftImg8bit_trainvaltest.zip`.'
unzip -n ${DATASET_DIR}/leftImg8bit_trainvaltest.zip -d ${DATASET_DIR}/

#echo "Removing `gtFine_trainvaltest.zip`."
#rm ${DATASET_DIR}/gtFine_trainvaltest.zip

#echo "Removing `leftImg8bit_trainvaltest.zip`."
#rm ${DATASET_DIR}/leftImg8bit_trainvaltest.zip

echo -e "\nFinished."
