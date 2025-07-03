# DatingMining

How do people meet and spark connections today? It’s lightning fast compared to the slow dance of our parents’ and grandparents’ era. You’ve heard the tales: meet someone, date for a while, propose, get married. Back in the day, especially in quaint small towns, you had basically one shot at love. No pressure, right? Mess up once, and that was it. Today, the real puzzle isn’t getting a date—it’s finding the *right* match amidst an ocean of options.

Over the past 20 years, dating has morphed from traditional approaches to online dating, then to speed dating, and now to online speed dating. Nowadays, you just simply swipe left or right, if that’s your vibe. We were curious about what factors during that brief interaction determine whether someone perceives the other person as a compatible partner.

Driven by this curiosity, our mission is clear: help you find a worthy match—or at the very least, spare you the agony of a mismatch. We build predictive models that can gaze into the crystal ball of compatibility, forecasting the chance of a **match** based on a bouquet of features—mostly describing individual quirks, interests, and all those little things that make us who we are. Originally, our target variable simply flagged if both parties wanted a second encounter. So, if you’re ever torn about giving that fleeting stranger a shot, well, our models might just nudge you in the right direction.

## About this Project

This project serves to demonstrate my acquired knowledge in the field of Data Mining through a relevant example. It originated as a team project during my Master's studies in Data Science at the **University of Mannheim** and has since been independently continued and expanded by me out of personal interest. Throughout the remainder of this project, I will continue to use the academic (editorial) "we."

## Dataset

The dataset used for this project originates from a speed-dating experiment conducted by **Columbia University** between 2002 and 2004. It contains data from **21 speed-dating sessions** involving mostly young adults meeting people of the opposite sex. Unfortunately, at the time of publishing this project, the dataset and its accompanying data key are no longer available at the original source: [http://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/](http://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/).

A total of 551 individuals participated, consisting of 277 men and 274 women. The dataset contains **8,378 individual observations**, with each row representing a speed date between two individuals and including **194 features**. Notably, the dataset only includes heterosexual pairings, which makes it somewhat outdated from a modern perspective.

## Data Cleansing

Besides its fundamentally interesting content, the dataset is also noteworthy for being extremely messy, making it ideal for applying various data cleansing tasks. The applied steps and further details about the dataset and its characteristics can be found in the notebook [`src/data_prep/data_cleansing.ipynb`](./src/data_prep/data_cleansing.ipynb).


## Data Preprocessing

The data preprocessing methods applied in this project are located in [`src/data_prep/data_preprocessing.py`](src/data_prep/data_preprocessing.py). These functions are designed to be executed within each fold of a cross-validation procedure, which is considered best practice to prevent data leakage between training and validation sets. Corresponding unit tests can be found in the [`unittests/data_prep`](unittests/data_prep) directory.
