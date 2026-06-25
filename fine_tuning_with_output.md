# Project Notebook: Yelp Dataset Review Generator


```python
!pip install datasets
```


```python
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
# Use a pipeline as a high-level helper
from datasets import load_dataset
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model,AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# !pip install datasets
from datasets import load_dataset # https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset

```

The Yelp Reviews dataset is available on [Hugging Face](https://huggingface.co/datasets/Yelp/yelp_review_full) and can be loaded using Pandas.

### Note!

The version of this dataset which we used for Homework 2 was "polarized" and used 0/1 label encoding for positive/negative sentiment. This is the original dataset which encodes labels using integers **0 through 4** to represent the corresponding star rating of each review (1 star to 5 stars) with 4 denoting the most positive sentiment and 0 denoting the most negative.

We hope that this more-granular labeling will allow for more project possibilities; if you wish to use 0/1 labels, feel free to define another label encoding and make it explicit in your project report (for example, mapping original labels 0-2 to 0 and mapping original labels 3-4 to 1).

The dataset is divided into train and test splits with 650k and 50k examples, respectively. For a validation set, we recommend subsetting from the training dataset.

Now you're ready to explore the dataset!


```python
# HYPERPARAMETERS
batch_size = 8
num_epochs = 5
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

    Using device: cpu



```python
def tokenize_function(examples):
    # print(examples)
    inputs =  tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs
```


```python
# Loading and Tokenizing the Dataset
dataset = load_dataset("yelp_review_full")

x = 5000
dataset_train = dataset['train'].select(range(x))
dataset_test = dataset['test'].select(range(x // 10))
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

tokenized_dataset_train = dataset_train.map(tokenize_function, batch_size = batch_size)
tokenized_dataset_test = dataset_test.map(tokenize_function, batch_size = batch_size)
```

# Fine tuning

This samples training and testing data from the overall dataset.


```python
# https://medium.com/@prashanth.ramanathan/fine-tuning-a-pre-trained-gpt-2-model-and-performing-inference-a-hands-on-guide-57c097a3b810

# https://huggingface.co/openai-community/gpt2#how-to-use
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
output_dir = '/mnt/results'
logging_dir = '/mnt/logs'
model_output_dir = '/mnt/results/model'

```


```python
# Define training arguments
output_dir = '/mnt/results'
logging_dir = '/mnt/logs'
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy='epoch',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=logging_dir,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
)

# Train the model
checkpoint = '/mnt/results/checkpoint-5000'
trainer.train(resume_from_checkpoint=checkpoint)
trainer.train()

# save the model and tokenizer explicitly
model_output_dir = '/mnt/results/model'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
```

# Trying our finetuned model


```python
def makeYelpPrompt(review, starRating, sentiment="serious"):
    return f'''\
Continue the following Yelp review which gives a rating of {starRating+1} stars and has a {sentiment} tone:
{review}\
'''
```


```python
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
set_seed(42)
```


```python
prompts = []
prompts.append(makeYelpPrompt("I am so disappointed in", 0)) # Negative prompt
prompts.append(makeYelpPrompt("I am in awe of", 4)) # Positive prompt
prompts.append(makeYelpPrompt("I am in awe of", 0)) # "Positive" prompt
prompts.append(makeYelpPrompt("This restaurant is", 0)) # Can GPT fill in sentiment based on star rating?
prompts.append(makeYelpPrompt("This restaurant is", 4)) # Can GPT fill in sentiment based on star rating?
prompts.append(makeYelpPrompt("This restaurant is", 2)) # Can GPT fill in sentiment based on star rating?
prompts.append(makeYelpPrompt("This restaurant is", 2, "funny")) # Can GPT fill in tone?
prompts.append(makeYelpPrompt("This restaurant is", 2, "optimistic")) # Can GPT fill in tone?
# Are longer initial reviews better?
prompts.append(makeYelpPrompt("Rose tea is great value for the price, especially for a college student. There’s usually a 10 min wait if you order at the counter so I suggest ordering ahead of time. The fried chicken over rice is", 4))
prompts.append(makeYelpPrompt("I used to enjoy this place...until I got food poisoning. I ordered a chicken & broccoli dish and believe it was the broccoli that was somehow contaminated. Haven't been back since. If you eat something from here that doesn't taste right on first bite, trust your gut (literally)", 0))
```


```python
for prompt in prompts:
    output = generator(prompt, max_length=50 if len(prompt) < 50 else 120, num_return_sequences=5)
    for d in output:
        print(d['generated_text'])
        print('-----------------------')
    print('~~~~~~~~~~~~~~~~~~')
```

    Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.


    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am so disappointed in my dining experience here. Not sure if this place is for me or if this is for food. The menu was ok and for sure, so was the meal.
    
    The service was not awesome, only getting the same menu and the same items. All that said, if I had to buy the $4 margaritas and only two drinks the order would probably be out of order. They're always open and open on weekends. If we were here in the summer and
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am so disappointed in this review. The delivery is horrible and the service is poor. It is not a place to go to shop or give my friends something to buy I would take this very seriously.
    
    Yelp's customer service is always the better when it comes to their service. They don't serve the food, have no complaints, etc. Not even close.
    
    The Yelp reviews come together to not take the bait when they come across it. It's not a place,
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am so disappointed in both the restaurant and review on this place! The food is good as it's mostly traditional. Would have gone with this again. The service went pretty well to my review. I did find the waitress to be more friendly and the meal was delicious. Had my steak with some fish, which was pretty much all that was left on me, I will definitely go again. Overall will not go back to this restaurant again!
    
    I like being able to order dinner before the restaurant
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am so disappointed in this place. I was looking at the reviews for several reviews and the product was very easy to understand, especially when I looked through the store's information and I was getting quite a few of the generic products. They put a sticker on the door asking whether you need to buy it or not and told me it is in stock. I could also go to my other online bookings online, and if I was lucky, I would find the product at the next store. I went
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am so disappointed in this restaurant. The food was bad but good. The drinks tasted terrible. No service even though it's pretty good. The menu was good and the prices came from a lot of money. I'm sure I will come back.
    
    I would not call this a place where you get drunk and are drunk all the time with only one of the options that I've seen. The food was great and the atmosphere was great. This place is not a lot more expensive than any
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    I am in awe of the product - excellent customer service and quality ingredients and amazing price for what you find. This is an excellent value to you in my opinion. I would highly recommend this chain but the reviews to others can be quite long and be unreliable.
    
    The Yelp reviews are great and there aren't numerous reviews of the service and product. It's a small area and not a typical place for new members, but the quality of service, service, products and great customer service is what you
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    I am in awe of our restaurant, and if you follow the Yelp reviews, we're sure you'll agree that we deserve more of them. I am just as happy to follow my friend's reviews as I am to keep one. For one, we've never had a single problem where we weren't satisfied with the restaurant (including with all the staff and our guests.) And for a large portion of the time, I've been able to stay in for food and drink a little bit longer, despite
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    I am in awe of the design. I love the color, a deep brown, dark olive color and a cool finish on the back of the headcaps and the white cap. I also love the way it comes off my person, and will be continuing to try. There are still a few small bumps in the side pieces though but I am looking forward to going through it. In particular, the top handle is pretty good, too, being a little too high to grip properly and even pushing against it
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    I am in awe of the great food and comfort of the food. It's not a deep fried egg roll, it's good and it's pretty. I wish that there was some more egg-covered fried chicken in there. I was expecting the fried chicken would have been more juicy but it was rather bland and bland. Overall I liked the service with this location as a location and as a family that eats here every day. Good food, good wine, and great seating, and just how the other
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    I am in awe of how many people can actually make this product in one day. I also don't think there are any better way to find the product to use all in one place, so don't try it in a mall. I am at home in my car waiting to purchase my next car. My wife likes to buy it at the car dealership and she has only purchased her car for 5 dollar parking ticket. I ordered 3 of the other cars based on their cost and I wasn't impressed.
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am in awe of the beautiful artwork at the store and my first try on the "Wearing Bikini" bra. I love how subtle it is made and will be ordering more soon. My husband and I took the first step back and were both very impressed. Very comfortable and had a great feel for the bra. The "Bikini" is very soft and well thought out. My husband was able to wear it as she liked. The "Lips" were very soft and held up even
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am in awe of this place. I wish they didn't have a list of every single one of their popular restaurants (I am a foodie, and am in need of an appetizer and so do the food staff!) And also because I'm so fond of D'Angelo!
    
    Food here is fine, but they're all too familiar. Everything you'd expect at D'Angelo! But with a small menu and an expensive tip they just don't be this bad. I'm a
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am in awe of how well this cocktail is made, not by the wait staff, but by the food which is great and amazing, especially the shrimp, cucumber, and blue sauce. I think it's pretty great to have them on hand to go take home to check the food that will be on offer.
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am in awe of the reviews and the staff, they really care about my quality and care-shares about my food. They always come with a great explanation, they've done more than anyone could ask for - they really care about my food, which is amazing. I have been ordering food from them the past few weeks. They were very accommodating, they made me feel good and I will definitely return to them again!
    
    I had the great news that my favorite food in Bali was
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I am in awe of this book and am absolutely blown away that I'm still reading it. It gives you the same information that we all need. You read it as you would an encyclopedia as you will soon begin to realize.
    If you like what you see, please let me know and I am willing to do my best to review the book, though all reviews are reviewed by an individual or company. I recommend you click through to the review when the book is in print. If you would like
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    This restaurant is so bad I wish I could eat here everyday
    This restaurant is just amazing. Every time I come there for breakfast I never even ask which way to get off the plane when I get on. Its so easy to get off plane and all the nice staff and happy people all around. That is a shame at what the food was so far. We tried the grilled steak in the sandwich, just ok so the steak is ok but it does sound really bland and bland. I could have been
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    This restaurant is very good. I'm sure I've only eaten one or two of them, but that's because it was always on the menu. Their beef was absolutely delicious and very tasty - it was the perfect complement to my chili. The only bad thing this place has? The customer service. I know it's not right that you would order from here by yourself but the one thing I will mention is that they are only on the menu. They have one of two tables on the patio with a
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    This restaurant is excellent. The food, service, and attentive servers were outstanding. We just ordered the Chicken and Onion Salad and as per their description, the chicken was ok but they may very well need a more creative and tasty option like shrimp or beef. The salad was ok. It looked really good as well. Just the two items were slightly mixed but overall good service. Highly recommended!
    
    This is a very small gem and one of the few Yelp reviews that I was able to read. The
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    This restaurant is great. Their food is good but they are always at their very best. Thank you place!!
    
    The food is mediocre. I've been coming here for years and I have never had a negative experience with them. They only come in 2 hours. The service at the service spot is pretty crappy as of late. It's very rude when you're seated for 5 minutes. There was a table at the front of the restaurant in my room. When my wife decided I wanted to check
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    This restaurant is delicious I am very disappointed with the service and restaurant. I ordered my first entrees at 4 weeks and was ok. My first bite was a big burrito which tasted good but wasnt really bad, I will definitely be back. Food: I've had almost every variety of entrees here and they are pretty good. I ate the entrees with black beans, cajun, guacamole, onion, bam mam, and a tomato. There was some meat
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    This restaurant is so good it's called "the best". It's a place where the kids don't drink a lot even though they are. I'm a member of a group of high school, college students who are going off to college due to the music and the food. They don't even know how to bring beer and don't even need a phone. To them, this is just a place to go get their drinks because I'm not one of them. The menu and service is pretty good
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    This restaurant is actually rather good, it can be an easy day and its food is really good, they have free wine (which is what the company recommends) and the customers are very friendly. They also have an awesome "SUNDAY MONDAY," which is pretty cool even though this time was much better. I've mentioned before about being a very busy and stressful business, and I'm really looking forward to the days of running home and the weekends. But I don't think there's a
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    This restaurant is a popular location on a small and cozy street that serves very good food. It's very well decorated with classic Mexican cuisine. While at the counter it was just a small place to sit and eat. I would recommend this place. Also have a special order of beans at the end: I think they offer hot dogs with sauce. It's amazing how much deliciousness there is. Also the food tastes like some of my favorite Italian cuisine, this way it gets more flavor and less cheese.
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    This restaurant is good, they have a great patio so good food, and the drinks are good.
    If it were a bit more of a restaurant in the park it would be a bit closer to home. The wait staff is not helpful, and a large amount of wait and what not. The food is bland with a little more freshness.
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    This restaurant is so good it was my go-to for any restaurant that was around the corner from my house. This is not, nor has been, where you'd expect it to be. There is some great food for any, good and priced. The only thing I would not recommend is the lack of service.
    We were sitting at 7 pm so we went to stay up late and waited about 30 minutes for some food to come by our table and we were there to stop by some people and
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 3 stars and has a serious tone:
    This restaurant is very crowded, with patrons waiting for several hours in front of a window to stare at the glass of wine; no idea what kind you're getting, or maybe they can pick up more wine? You go down and try some of my famous dishes (I promise you can't miss one; it's like you've been on all 5 of them). Of course there is no need to order and there are no alcohol at the bar with your order. At first you get the impression that the
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a serious tone:
    This restaurant is very good. The staff seem professional and friendly. My good friend will be back there soon to check out the place again!
    
    I will be back again. Not being a regular diner, it is actually kinda lame to stay on an empty stomach. They always try to remind me why I come here and where I went instead.
    
    My favorite bar in San Jose! What better place to go than with a bar that is a huge hit with the San Jose area. This
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a serious tone:
    This restaurant is very good. The chef's food is high quality - everything you pay for tastes exactly the way it should and the food looks clean...the owners were willing to give us a lot of time due to their customer service and service on a daily basis. They had a nice mix of old fashioned old school, good old fashioned, and new new things. The seating is comfortable, the food is good, and the service was great. A bit of an issue though. If you are having a
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a serious tone:
    This restaurant is not available here in San Francisco. We have never considered renting a bar here. We are not a bar. We did not ask you to leave and have asked for your help. We apologize for your disappointment. We'll do our best to fix this once we have resolved the issue.
    Here are some reviews of the bar I went to:
    I was shocked at the fact that this location does not offer a drink menu. No matter what I wanted to order you would not be able
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a serious tone:
    This restaurant is a popular destination for American families looking to spend some time with an authentic American food. One day this wonderful American food will provide a great meal. It just makes you smile more when you see good food. A great casual atmosphere and a well laid back setting make this restaurant an ideal place to relax for a quick snack. At least this restaurant had a good service. The food itself is amazing in the evening and even the food is very well done. The service and the comfort of the service
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 3 stars and has a funny tone:
    This restaurant is just great for a cool place for kids to go to and all of them can enjoy a nice meal. I went for the steak but it was so good the chicken was just so good. The menu is well-made as well and you should be able to find one in most parts of Japan (unless you are in San Francisco). If you are lucky enough to find one on a Saturday afternoon try the chicken pong. The pong has lots of veggies from local to foreign, so
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a funny tone:
    This restaurant is very accommodating. I would recommend this restaurant. 2 people found this review helpful.
    
    A wonderful place. Very comfortable. Friendly customer service
    
    Excellent food service and friendly waitress. Was able to order a portion of the lamb, sardines, and all of our pastries. Very easy to reach, and you can order the meat when needed and you won\'t have to wait, and if you order it online, you will be able to request the dish.
    
    
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a funny tone:
    This restaurant is definitely overrated but still very good The food and atmosphere is pretty good and nice service is great
    In my first stay I did have a plate of my favorite pizza and my wife wanted something a bit different in it. She had it for dinner and it was very good. She kept it a day or two to try the hot pepper poblano chili and there was enough for a large plate. The only complaint is that the toppings are bland. On a Friday and Saturday, when
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a funny tone:
    This restaurant is so clean. This is very clean, even in the sun, and not that dirty. For comparison, the other places that I've visited the past few months have been pretty clean... the most I've seen is at the entrance of the building. Just not clean, and not that clean. The owner is a lovely person who respects all of you. The first person I spoke to said that he really likes this place... and that is as well. This restaurant is so clean. This
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a funny tone:
    This restaurant is very nice, but they serve a combination of the worst and best. This may be the place to look, but with all of the wait staff and service, you had little chance to ask for a place not even that bad. But they left all of us seated, and the wait staff was still there to pick up the pace. The restaurant's name is a bit of a joke at first glance, but after being told by the wait staff "don't do this kind of business"
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 3 stars and has a optimistic tone:
    This restaurant is open 9-5 Mondays thru Wednesdays 9am-1:30pm.
    The service is not fast in all other respects. My husband does not like the sound. He is used to being able to order food and have it wait in the back room when our food arrives. He goes to eat, and he ends up eating at another diner, the same diner. The waitress has already spoken by phone with his supervisor. He does very little wrong after talking with her. I
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a optimistic tone:
    This restaurant is very clean, clean, and really tasty! Nice staff and attentive staff. Service is fine, and the food is tasty. Can't say enough good things about the place....It has a friendly and professional staff and is totally friendly. Definitely recommended!
    
    This has always been a must try spot for my little sister. The reviews are super nice. She is completely in love with this place.. I went from a big favorite to love and enjoyed it. This is the restaurant I will
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a optimistic tone:
    This restaurant is a wonderful place to get dinner, as they call themselves "a great food truck". They are located in a small town area close to the Kish River and have a nice selection of fine restaurants that I'll have to try from now on. If you aren't sure if you have a spot for them, this is a good option. The best part? Their website is a lot more informative than I expect, I mean the company lists their reviews with each of their menus (with links
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a optimistic tone:
    This restaurant is packed for the geeks out there. Food here. They are so good with this salad/pepper/pork and avocado combo, the only ones in town are the kids, my older brother, and my dad. The steak and salad were pretty good and I couldn't keep my mind off of this. I will definitely be back for another quick bite.
    
    I wanted to make this for home plate, but here's how it went out. My friend had her recipe and
    -----------------------
    Continue the following Yelp review which gives a rating of 3 stars and has a optimistic tone:
    This restaurant is beautiful and authentic. The kitchen is very basic and easy to manage. I'm going to order a pizza from the menu - I don't have the whole pizza but it's good and has a healthy portion. The food that I was craving is extremely clean and delicious. The atmosphere is very laid back and pretty. There are many large rooms with full service restaurant as well. I had to make a few changes after the show. I tried using a new thermostat - that's just
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    Rose tea is great value for the price, especially for a college student. There’s usually a 10 min wait if you order at the counter so I suggest ordering ahead of time. The fried chicken over rice is really delicious and so will your food. I got to say that the flavor is very hot and cold like I would for most fried chicken. I wish I had said so. The only other thing i would add is water to make the taste more authentic since the water will just make your
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    Rose tea is great value for the price, especially for a college student. There’s usually a 10 min wait if you order at the counter so I suggest ordering ahead of time. The fried chicken over rice is great too, the service is excellent, and the staff is very friendly. The water is good so don't go overboard. The only downside is if you leave the water cold you go crazy and end up with a leak for the next two hours (I was going to leave it for
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    Rose tea is great value for the price, especially for a college student. There’s usually a 10 min wait if you order at the counter so I suggest ordering ahead of time. The fried chicken over rice is good too, although no beef jerky (or fried pork) on top of the fries. The other items I was really looking forward to were the chile peppers, onion, and chili peppers. Overall my review is a little light, but I hope others did like it. I
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    Rose tea is great value for the price, especially for a college student. There’s usually a 10 min wait if you order at the counter so I suggest ordering ahead of time. The fried chicken over rice is a great experience but I did pass the waiter but it's also worth it for his quick service and quality. Just take a look at this restaurant if you want authentic Japanese. I'll be back :)
    
    I've been to several reviews for this place - so I knew what to
    -----------------------
    Continue the following Yelp review which gives a rating of 5 stars and has a serious tone:
    Rose tea is great value for the price, especially for a college student. There’s usually a 10 min wait if you order at the counter so I suggest ordering ahead of time. The fried chicken over rice is good, as is the other chicken dishes that may be in the dishwasher/room temp. The rice is tasty.‡‡‡" But even from a 10-minute counter visit, it was surprisingly warm and quite pleasant. I would leave the bathroom while waiting for
    -----------------------
    ~~~~~~~~~~~~~~~~~~
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I used to enjoy this place...until I got food poisoning. I ordered a chicken & broccoli dish and believe it was the broccoli that was somehow contaminated. Haven't been back since. If you eat something from here that doesn't taste right on first bite, trust your gut (literally) and you deserve a second bite. My sister used to order the chicken and broccoli dish at least once a day. But as a whole I don't see a problem with it. Also, I had the chicken in
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I used to enjoy this place...until I got food poisoning. I ordered a chicken & broccoli dish and believe it was the broccoli that was somehow contaminated. Haven't been back since. If you eat something from here that doesn't taste right on first bite, trust your gut (literally) and be prepared to be disappointed. As you start to eat the chicken & broccoli, the chicken gets chewy. The broccoli is almost burnt (which I know to be bad). I'd still consider throwing it out
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I used to enjoy this place...until I got food poisoning. I ordered a chicken & broccoli dish and believe it was the broccoli that was somehow contaminated. Haven't been back since. If you eat something from here that doesn't taste right on first bite, trust your gut (literally)...and don't put a spoon to the wrong side of the dish. I usually give a good one a 4 out of 5 because this is such a tiny establishment. As a second review by Dany says they
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I used to enjoy this place...until I got food poisoning. I ordered a chicken & broccoli dish and believe it was the broccoli that was somehow contaminated. Haven't been back since. If you eat something from here that doesn't taste right on first bite, trust your gut (literally) as you will get the unpleasant taste of boiled cauliflower. My only thing I did to try this place and am glad I did. Food was good. The chicken was amazing!
    -----------------------
    Continue the following Yelp review which gives a rating of 1 stars and has a serious tone:
    I used to enjoy this place...until I got food poisoning. I ordered a chicken & broccoli dish and believe it was the broccoli that was somehow contaminated. Haven't been back since. If you eat something from here that doesn't taste right on first bite, trust your gut (literally) and say to yourself: there is nothing wrong with making a new dinner every day without it being perfectly good. If you want to learn the difference between good and bad, go down this street, it's going to
    -----------------------
    ~~~~~~~~~~~~~~~~~~




# Generating reviews from test data


```python
dataset_test_large = dataset['test'].select(range(x // 10))
```


```python
length = 10
test_prompts = dataset_test_large.map(lambda x: {'prompt': makeYelpPrompt(' '.join(x['text'].split()[:length]), x['label'])})
# test_prompts_large = dataset_test_large.map(lambda x: {'prompt': makeYelpPrompt(' '.join(x['text'].split()[:length]), x['label'])})
```


```python
output = generator(test_prompts['prompt'], max_length=75)
temp = test_prompts.add_column('output', [L[0]['generated_text'] for L in output])
test_prompts = temp.map(lambda x: {'generated_output' : ':'.join(x['output'].split(':')[1:])})
test_prompts = test_prompts.remove_columns('output')
```

# Using classifiers to quantify performance


```python
from transformers import pipeline
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# We will be using BCE loss to measure the performance of our fine-tuned transformer against these models.

loss_fn = nn.BCELoss()

```

    Using device: cpu


This classifier, found at https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment, is designed for product reviews. It maps an input text to a probability distribution on $\Omega=\{1,2,3,4,5\}$, which is the number of stars.


```python
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device="cuda",return_all_scores=True)

one_hots = nn.functional.one_hot(torch.tensor([0,1,2,3,4]), num_classes=5).float()
# Using cross entropy loss from the 1-hot vector on expected_sentiment with the classifier's output on review
def calculate_review_loss(review, expected_sentiment):
  res = pipe(review)

  # turn result into probability distribution
  distribution = torch.tensor([x['score'] for x in res[0]])

  # alternative: use gaussian based assumption instead of one-hot
  # sigma = .5
  # class_indices = torch.arange(5, dtype=torch.float32)
  # gaussian = torch.exp(-((class_indices - (expected_sentiment)) ** 2) / (2 * sigma ** 2))
  # gaussian /= gaussian.sum()

  # return BCE loss
  return loss_fn(distribution, one_hots[expected_sentiment])

print(calculate_review_loss("Rose tea is possibly the very best thing I have ever tasted.", 4))

```

This classifier, found at https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student, is designed for product reviews. It maps an input text to a probability distribution on $\Omega=\{-1,0,1\}$, which indicates negative, neutral, and positive sentiment.

Finally, this third classifier, found at https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest, was trained on tweets. It maps an input text to a probability distribution on $\Omega=\{0,1,2\}$, which indicates negative, neutral, and positive sentiment respectively.


```python
test_prompts = test_prompts.map(lambda x: {'review_loss' : calculate_review_loss(x['generated_output'], x['label'])})
```


```python
import numpy as np
import matplotlib.pyplot as plt

plt.hist((test_prompts['review_loss']), bins=30, alpha=0.6, color='b')

# Add titles and labels
plt.title("Distribution")
plt.xlabel("Binary Cross-entropy loss")
plt.ylabel("Count")

# Show the plot
plt.show()

```


```python
print(sum(test_prompts['review_loss']) / len(test_prompts['review_loss']))
```


```python
import json
with open('/content/generated_outputs.json', 'w') as f:
    json.dump(test_prompts.to_dict(), f,indent=4)
```
