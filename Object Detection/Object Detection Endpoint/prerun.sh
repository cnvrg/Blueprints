#!/bin/bash
apt-get install -y libgl1-mesa-dev
apt-get install -y libfreetype6-dev
apt-get install -y libmagic1
pip uninstall -y pillow
pip install --no-cache-dir pillow