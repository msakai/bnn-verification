#!/usr/bin/bash

pandoc \
    --standalone \
    --toc-depth 2 \
    --shift-heading-level-by=0 \
    --metadata title='BNN Verification Instances for the MIPLIB 2024 submission' \
    --template=easy-pandoc-templates/html/easy_template.html \
    --fail-if-warnings \
    --output=additional_files/README.html \
    additional_files/README.md
