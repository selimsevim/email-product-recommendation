# Progressive Hybrid Product Recommendation for Emails 
This repository implements a progressive recommendation system that combines a cold-start collaborative filtering model with online learning and persistent user queues. The system is designed to adapt over time by incorporating new browsing events (views, carts, removals, and purchases) while preserving historical knowledge and enforcing business rules—such as not recommending or storing purchased products.

## Overview
This system consists of two major components:

1. Training (Cold-Start):
	- Uses historical browsing data to build a collaborative filtering model and create user and product mappings.
2. Inference and Online Updates:
	- Loads the cold model and applies online updates with new (delta) data.
	- Generates personalized recommendations and maintains a persistent recommendation queue for each user.
	- Enforces rules so that purchased products are never recommended or stored.

## Dataset
This project uses the E-commerce Events History in Cosmetics Shop dataset from Kaggle. You can download the dataset from the following link:
https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop

### Key details about the dataset:

- **Domain:** E-commerce events in a cosmetics shop.
- **Data Coverage:** The dataset includes historical event data capturing various user interactions.
- **Event Types:**

	1. **View:** Indicates a product was viewed.
	2. **Cart:** Indicates a product was added to the shopping cart.
	3. **Remove_from_cart:** Indicates a product was removed from the cart.
	4. **Purchase:** Indicates a product was purchased.

- **Key Columns:**
	- **user_id**: Identifier for the user.
	- **product_id**: Identifier for the product.
	- **event_time**: Timestamp for when the event occurred.
	- **event_type**: Type of event (e.g., view, cart, purchase).

- **Usage:**- 
	The dataset is used to simulate real-world user interactions. Historical events are processed for cold-start model training, and new (delta) events are used for online updates during inference, ensuring that recommendations remain relevant over time.

This dataset provides a realistic foundation for building and testing dynamic recommendation systems in an e-commerce context.

## Architecture and Data Flow

|  Component |  Description |
| ------------ | ------------ |
|  Data Sources	 |  Browsing CSV files with user interactions and timestamps. |
| Historical Data	  |  Used by **training.py** to build the cold model and mappings.|
| Delta Data	  | New events detected since the last update, used for online training in **inference.py.**|
| Model Updates	  | Online training updates the model weights, which are then saved for future inferences.  |
|  Persistent Queue	 | A SQLite database (**recommendation.db**) stores user-specific recommendation pools (queues).  |

## Model and Recommendation Logic

### Collaborative Filtering Model

**- User and Product Embeddings:**
The model learns embeddings for users and products.

**- Fully Connected Layers:**
Concatenated embeddings are passed through a series of layers to produce a purchase probability.

**- Recommendation Function:**
Scores all products for a given user and returns the top-K recommendations.

### Predefined Scoring Rules

| Event  |  Score Change |
| ------------ | ------------ |
| View  |  +1 |
| Cart  | +3  |
|  Remove from Cart |  -1 |
| Purchase	  |  (Used to remove items from pool) |

## Persistent User Queue and Database

- Database:
A SQLite database (recommendation.db) stores each user's recommendation queue in JSON format.
- Queue Format:
Each queue is a dictionary with a key "pool" that holds a list of items:
[productID, counter, score]
- Rules Enforced:
	- Do Not Recommend Purchased Products
	- Do Not Add Purchased Products to the Queue
	- Remove Purchased Products from the Queue

## Predefined Rules and Fallbacks

- **Cleaning Process:**
Purchased products are filtered out from the persistent queue.
- **Fallback Candidates:**
If the cleaned queue is too small (e.g., fewer than 6 items), fallback candidates are generated using:
	- Global popular items (from purchase counts).
	- Collaborative filtering and category-based popular items (filtered to exclude purchased items).
- **Rotation Logic:**
Recommended items have their counter incremented and are rotated to the end of the queue so that the user sees fresh items on subsequent runs.

## Training Process
The **training.py** script:

- Data Filtering:
Loads historical browsing data, filtering by a target number of unique users.
- Mapping Creation:
Creates contiguous mappings for user and product IDs.
- Model Training:
Trains a collaborative filtering model using labeled events (purchase = 1, non-purchase = 0).
- Persistence:
Saves the cold model weights (trained_model.pt), and the user and product mappings (user_mapping.pkl, product_mapping.pkl).

## Inference and Online Updates
The **inference.py** script:

- Model Loading:
Loads the cold model and mappings.
- Delta Data Detection:
Checks for new browsing data (delta data) since the last update.
- Online Training Update:
If new delta data is available, the model is updated online and the new weights are saved.
- User-Specific Queue Processing:
	- Loads the persistent queue for each user.
	- Removes any purchased products from the queue.
	- Merges the queue with newly computed candidate scores.
	- Applies fallback and rotation logic to ensure fresh recommendations.
- Output:
Final recommendations and updated queues are saved (e.g., CSV outputs for recommendations and debug activity).

## Usage
1. **Training:**

Run the training script to generate the initial cold model and mappings:
```bash
python training.py
```
2. **Inference and Online Updates:**

Run the inference script to update the model with new delta data and generate recommendations:
```bash
python inference.py
```
3. **Data Files:**
   
Place your browsing data CSV files in the BrowsingData folder.

4. **Database:**

The system creates/updates recommendation.db automatically.

## Considerations and Future Improvements
- **Model Drift:**
Monitor performance to prevent drift due to continuous online updates. Consider periodic retraining from scratch if necessary.
- **User-Specific Policies:**
Fine-tune rules for handling purchased items or for tailoring fallback logic.
- **Fallback Strategies:**
Additional fallback mechanisms (e.g., content-based filtering) could further enhance recommendation quality.
- **Logging and Monitoring:**
Implement more detailed logging for production monitoring and debugging.

