# GeoEnrich-NYC — four-page NeurIPS blueprint

## Title

**GeoEnrich-NYC: A Geospatial Text-Tabular Dataset for Airbnb Price Prediction**

## Abstract

We provide a framework to enrich any dataset with existing latitude-longitude coordinates,
where a regression task might benefit from a descriptive summary of each coordinates' surroundings.

## 1. Introduction

Accommodation datasets (Airbnb, booking.com) often include latitude-longitude coordinates
of listings.
The location of a listing can be leveraged as a significant indicator for price prediction
regression tasks, though it is unlikely that using the coordinates as-is will be an informative
feature for regression models.
To solve this problem, traditional ML regression models perform feature engineering and use
techniques such as HDBSCAN [QUOTE] clustering, in order to leverage this potentially informative feature.

In addition, the surroundings of a listing can often provide signal regarding it's price. For example:
It is reasonable to assume [QUOTE] that Airbnb listings in downtown Manhattan, where there is a high density
of restaurants tourist attractions, will be pricier than listings in the outskirts of the city - where little
to no amenities can be found.

By following these assumptions, we hypothesize the following: a natural-language description of a listing's
surroundings will improve models' fit in a nightly-price prediction regression task.

To achieve this, we leverage the given latitude-longitude coordinates from an existing Airbnb listings dataset
and enrich them with geospatial data by querying the Overture API [QUOTE] and enriching each listings with
business density information of it's surroundings. We then use batch LLM API calls [QUOTE] to transform the
numerical density information into a description of each listing's surroundings.

## 2. Related work

**Text-tabular curation and target-aware representations.**  The goal of this project
is curating a dataset for the MulTaBench framework. MulTaBench frames text--tabular datasets
as candidates for curation rather than as evidence that any
single model is accurate: joint frozen inputs must improve on both unimodal
baselines, and target-aware representations (TAR) must improve on frozen ones
across several learners [QUOTE].  Its text protocol begins with a
frozen pretrained encoder, such as E5 (Wang et al., 2022), and adapts the encoder
to the prediction target with low-rank updates (LoRA; Hu et al., 2022).  We adopt
that protocol unchanged.  GeoEnrich-NYC contributes a new dataset candidate and
its reproducible curation pipeline; it neither proposes a tabular learner nor
claims that LLM verbalization is a replacement for target-aware representation
learning.

**Geographic context in Airbnb pricing.**  The basis of our approach is the idea that
geographic context provides a strong signal for nightly-price regression models.
In a personal project [QUOTE][airbnb-regression github] I've used a similar dataset,
and performed feature engineering, transforming the raw coordinates into neighbourhoods,
derived using HDBSCAN clustering, and neighbourhood-median-price encoding as a feature.
This has proved to perform better than a city-wide encoding, which provides the base hypothesis
that a surroundings summary might provide similar value.
Similar approaches were applied in [QUOTE] which achieved SOTA for this specific regression task,
and is generally common practice.

**Prior Airbnb price studies model.**
location directly through spatial weighting, coordinates, local amenities, or
spatial effects.  For example, Zhang et al. (2017) use a geographically weighted
approach to study spatially varying price determinants, and Tang et al. (2019)
model spatial effects in peer-to-peer accommodation prices.  These studies support
the relevance of local context, but they do not test the MulTaBench question posed
here.  GeoEnrich-NYC instead uses coordinates only to construct evidence, removes
them from model inputs, and exposes a constrained textual account of nearby places
and landmarks.  This design tests a coarsened representation of location; it does
not estimate causal amenity effects or solve spatial dependence.

**LLM enrichment.** [FILL] Previous dataset curation studies have leverages modern LLMs
both to structure unstructured or semi-structured data [QUOTE] and to generate
a natural-language description of various features [QUOTE]. These techniques
provide the basics upon which our solution is built.

## 3. Dataset origins and curation methodology

### 3.1 Dataset and sources

Inside Airbnb [QUOTE] is a project that provides quarterly data and advocacy
about airbnb's impact on residential communities, it's data is freely available
online. We used their dataset [QUOTE][date, location] for the evaluation pipeline,
though the same pipeline can be applied to any other of their provided datasets.

Overture Maps Foundation is a non-profit organization that provides free and open-source maps data.
The Overture Maps API [QUOTE] is a RESTful API providing easy access to the geospatial data, targeted
towards web developers. We used this API to query GeoJSON data of businesses and landmarks of our
desired city, and performed a join by proximity for each listing.

### 3.2 pipeline

**Data cleaning.** After acquiring the original dataset from "Inside Airbnb", we perform a standard
feature engineering pipeline: columns that do not partake in the regression task are dropped,
the schema is defined and enforced, features such as "number of rooms" and "numbers of baths"
are extracted from semi-structured text fields using simple regex rules. Listings with missing or invalid prices are dropped.
In addition, the target nightly-price column of the final version of the dataset is log-transformed -$\text{target} = \log(1 + price)$.
We were left with [Y] rows, each with [X] numerical and categorical columns. [The schema and features distribution plots can be found in Appendix A].

**Geospatial enrichment.** We query Overture API for all data within the city polygon and save it locally.
We apply a join such that each listing coordinate is assigned all POIs (points of interest) in a 400 meter radius, saved in JSON format.
The original latitude-longitude columns are dropped.

**Aggregation.** each JSON columns describing nearby POIs is reduced down to a simple count,
according to a list of predefined buckets (for example: restaurants, parks, transportation, nightclubs, etc.)
Those categories are curated from the official Overture hierarchical taxonomy.
We calculate a global mean of each category, and replace the count-buckets of each listing with it's relative quantile.
Popular POIs (landmarks) such as The Empire State Building, Statue of Liberty and other tourist attractions are saved
and not reduced down to their categories, as they potentially carry stronger neighbourhood signal.

**LLM descriptive summary.** By the end of the aggregation stage, each row has a JSON object containing the listing's
relative density of each category in their 400 meters surrounding radius and a list of popular tourist attractions nearby,
including their exact distance. We take a prompt blueprint [APPENDIX] and structure the JSON summaries, preparing them for an LLM API call.
We send a batch request to an LLM, processing all 10k rows simultaneously and independently - The LLMs are prompted to describe
the surroundings, and are not given the actual coordinates, exact counts, or listing price, to prevent data leakage.

Multiple LLMs were tested on subsets of the dataset, the model chosen is Gemini-3.5-flash-lite [QUOTE]. It was chosen for it's relative
lack of hallucinations, generation speed, batch requests availability and lack of reasoning - as the price-per-row steeply increased for
reasoning models. A full model comparison is not in the scope of this project.

This pipeline results in [X] numerical columns and 1 textual column, describing a listing's surroundings.
Statistics regarding the text fields can be found in [APPENDIX]

[FIGURE: how a row turns into a description]

## 4. Results

The process involved trial and error with numerous prompt versions,
tested on a small subset ($\approx 2000$ rows) of the dataset, until we settled on the 16th variant [APPENDIX]
Once the final variant has been chosen, we have added surrounding descriptions to the entire [X] rows.
We then ran the official "MulTaBench" [CHECK] curation pipeline twice: with raw price and with log-transformed price
as the target column.

[TABLE: results for log-transformed and raw price. R^2, -RMSE, passing learners]

## 5. Conclusion, limitations, and future work

### 5.1 Outcomes

We have successfully passed the MulTaBench curation pipeline, passing 5/5 learners on joint signal,
and 3/5 learners on TAR signal.
The constructed framework is flexible, and could potentially enrich any number of datasets, provided that:
coordinates are already provided, and the regression / classification task could potentially benefit from a
description of the coordinates surroundings.

### Limitations

**Evaluation.** Though the evaluation pipeline passes successfully, the reason behind that could be further explored
It is unclear what the main driver of the price-signal is in each listing's surroundings description.
It is possible that it is derived by the explicit naming of popular landmarks, or parhaps

**Application.** Though the provided framework is flexible and could potentially be applied to any number of datasets,
the structure of the prompts make it more applicable for hospitality or real-estate related datasets.
In addition, the landmarks were curated manually and so applying the pipeline to a different city will require
updating that list, or perhaps automating that choice through some different API call.

### 5.3 Future work

This project only tackles a single city, while the pipeline can potentially be applied across all datasets with regression tasks
that could benefit from geospatial information. Future work could evaluate the robustness of this pipeline across datasets outside of
hospitality sector price regression tasks.

A major challenge of this project was increasing the variance of responses (measured with [FILL] while maintaining the TAR [CHECK] signal.
Future work could tackle increasing variance, by using several prompt blueprints, randomly assigning dataset rows to different LLMs, or
using more powerful models.

## References

## Appendix — tables and metrics only

Do not add prose, qualitative examples, figures, or new claims. Every appendix table must support a main-paper claim or artifact audit.

| Table | Required content |
| --- | --- |
| A1 | Complete data card: fields, types, units, target processing, attrition, missingness, duplicates, subgroup coverage |
| A2 | Source/provenance/reuse matrix: URL, snapshot, retrieval date, terms/license, transformations, redistribution status |
| A3 | Reproducibility matrix: folds, seeds, software, hyperparameters, PCA/encoder/LoRA, runtime, hardware, compute/environmental estimate |
| A4 | Full per-fold primary results, deltas, uncertainty, and pass flags |
| A5 | Raw-aggregate, genuine-text, mismatched-text, named-landmark, and redacted-landmark diagnostic results |
| A6 | Spatial-holdout construction, sample counts, and results |
| A7 | Grounding/privacy audit counts, named-versus-redacted re-identification checks, and artifact manifest |
