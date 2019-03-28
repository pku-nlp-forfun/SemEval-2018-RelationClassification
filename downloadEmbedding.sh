#!/bin/bash

origin_embedding=$1

echo "Downloading ${origin_embedding}"
wget "https://cloud.dfki.de/owncloud/index.php/s/WKOCMj5UYiSVZeR/download?path=%2F&files=abstracts-dblp-semeval2018.wcs.txt.gz" -O ${origin_embedding}.gz

echo "Extracting ${origin_embedding}"
gunzip -v ${origin_embedding}
