#!/usr/bin/env bash
# Download every dataset used in the project into data/.
set -e
mkdir -p data && cd data
# SMD (Su et al., 2019), released with OmniAnomaly
[ -d OmniAnomaly ] || git clone --depth 1 https://github.com/NetManAIOps/OmniAnomaly.git
# SKAB (Katser & Kozitsin, 2020)
[ -d SKAB ] || git clone --depth 1 https://github.com/waico/SKAB.git
# UCI Individual Household Electric Power Consumption (Hebrail & Berard, 2012), CC BY 4.0
if [ ! -f household_power_consumption.txt ]; then
  curl -L -o hpc.zip "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip" \
  || curl -L -o hpc.zip "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip"
  unzip -o -q hpc.zip && rm hpc.zip
fi
echo "md5 of UCI file (expected e05286cd4a257d80c3f0407b350df7dd):"
md5sum household_power_consumption.txt 2>/dev/null || md5 household_power_consumption.txt
