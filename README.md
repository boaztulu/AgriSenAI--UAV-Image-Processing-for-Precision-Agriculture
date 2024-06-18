griSenAI
AgriSenAI is a Python-based desktop application developed to automate the processing and analysis of UAV thermal and multispectral images for precision agriculture. This tool supports agricultural professionals by providing critical insights for data-driven irrigation and crop management decisions.

Features
Automated Image Processing: Extracts canopy temperature and vegetation indices from UAV thermal and multispectral images.
Geospatial Analysis: Accurate field and plot extraction, plant detection, and noise removal.
User-Friendly Interface: Developed with PyQt5, providing an intuitive and accessible interface.
Data Export: Outputs processed data into CSV files for easy analysis and decision-making.
Motivation
Unmanned Aerial Vehicles (UAVs) have become essential in agriculture for monitoring crops with high spatial and temporal resolution. AgriSenAI addresses the challenges of processing and analyzing large-scale UAV imagery by automating these tasks, thus enhancing the efficiency and accuracy of precision agriculture practices.

Installation
Clone the repository:

sh
Copy code
git clone https://github.com/yourusername/AgriSenAI.git
cd AgriSenAI
Install the required dependencies:

sh
Copy code
pip install -r requirements.txt
Run the application:

sh
Copy code
python agriSenAI.py
Usage
Import Data: Load thermal and RGB raster files.
Field Extraction: Automatically delineate field boundaries within imported images.
Plot Extraction: Identify and extract individual experimental plots.
Plant Detection: Detect plant areas using the Red-Green Ratio Index (RGRI) and Otsu's thresholding method.
Noise Reduction: Refine thermal data by reducing noise.
Data Export: Export processed data to CSV for further analysis.
Example
A comprehensive experiment was conducted at the University of Florida's Tropical Research and Education Center (TREC) to validate AgriSenAI. The app successfully processed datasets collected over three years from 24 plots under green beans and sweet corn, demonstrating its capability to handle large-scale remote sensing data.

Contributing
We welcome contributions to enhance AgriSenAI. Please fork the repository and submit a pull request with your improvements.

License
AgriSenAI is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

Acknowledgments
Funding for this research was provided by the National Institute of Food and Agriculture, U.S. Department of Agriculture, under award number 2020-67021-31965. The authors also thank the Water Resource Research Group lab at the Tropical Research and Education Center, Institute of Food and Agricultural Sciences, University of Florida, for their invaluable assistance.
