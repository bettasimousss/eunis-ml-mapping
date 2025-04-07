# EUNIS Habitat Maps: Enhancing Thematic and Spatial Resolution for Europe through Machine Learning

This repository contains the code and workflows used to generate the **EUNIS Habitat Maps** dataset, a machine learning-based enhancement of the existing EUNIS habitat classification across Europe. The pipeline increases both **spatial** and **thematic resolution**, leveraging remote sensing data and advanced classification techniques.

📄 [View the dataset on Zenodo](https://zenodo.org/records/11108226)


🌍 **Map Viewer**: You can explore the generated habitat maps interactively through the following map viewer:  
[**EUNIS Habitat Map Viewer (100m resolution)**](https://sarasi-moussi.users.earthengine.app/view/eunishabitats100m)


## 📜 Summary

The **EUNIS habitat classification** is essential for categorising European habitats, supporting European policy on nature conservation, and implementing the **Nature Restoration Law**. To meet the growing demand for detailed and accurate habitat information, we provide spatial predictions for **260+ EUNIS habitat types at EUNIS level 3**, along with validation and uncertainty analyses.

Using **ensemble machine learning models**, combined with **high-resolution satellite imagery** and other **climatic**, **terrain**, and **soil variables**, we produced a European habitat map at a **100-meter resolution**. This map indicates the most likely EUNIS habitat at level 3 for every location across Europe. Predictions were validated for three independent countries: **France**, **the Netherlands**, and **Austria**. We also provide information on uncertainty and the most probable habitats at level 3 within each **EUNIS level 1 formation**. 

This product is likely to be particularly useful for **restoration** and **conservation** purposes, and can be further refined with accurate, local **land cover data**.


## 🧠 Key Features

The EUNIS Habitat Maps aim to provide improved habitat classifications across Europe by:
- Enhancing the thematic resolution of habitat classes: at EUNIS level 3
- High spatial resolution over Europe (100m)
- Uses satellite derived biodiversity products (landscape composition, vegetation structure, vegetation phenology) and ancillary environmental datasets: climate, soil, topography, hyrography
- Uses machine learning models for habitat classification within each EUNIS formation 
- Exploits expert derived cross-walk rules for wall-to-wall habitat mapping


## 🛠️ Setup

To set up the required environment for this project, you can use Conda. Simply create a new environment by running:

```bash
conda env create -f environment.yml

```

## 🤝 Acknowledgements
This work was conducted within the EO4DIVERSITY project, funded by the European Space Agency (ESA) through its Biodiversity+ precursors programme: https://www.eo4diversity.info/


## 📚 Citation
Si-Moussi, S., Hennekens, S., Mucher, S., & THUILLER, W. (2024). EUNIS Habitat Maps: Enhancing Thematic and Spatial Resolution for Europe through Machine Learning (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11108226


## 📬 Contact
For questions, contributions, or feedback, feel free to reach out to us by creating an issue on GitHub.
 