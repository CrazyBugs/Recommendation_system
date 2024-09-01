import numpy as np
import pandas as pd
import marqo
import streamlit as st
from openai import OpenAI
import os


marqo_api_key_a = os.getenv('marqo_api_key')
openai_api_key_a = os.getenv('openai_api_key')

mq = marqo.Client(
    "https://api.marqo.ai",
    api_key=marqo_api_key_a
)
client = OpenAI(api_key=openai_api_key_a)

brands = ['Aqualogica','Aroma Magic','Beauty of Joseon','Be Bodywise','belif','Biore','Brinton','Cetaphil','Clinique','Colorbar','Conscious Chemist','Cos-IQ','COSRX','Dermalogica','Dot & Key',"Dr. Sheth's",'Earth Rhythm','Neutrogena','The Derma Co','Minimalist',"Re'equil",'Lotus Herbals','Biotique','Lakme','Plum','Wishcare','Fixderma','La Shield','Mamaearth','Lotus Professional','Foxtale','Lacto Calamine','SunScoop','Innisfree','Garnier','VLCC',"L'Oreal Paris",'ISDIN','Riyo Herbs','Ponds','Forest Essentials','NIVEA','Himalaya','Bioderma','Hyphen','Iba','Kaya Youth','Moha','WOW','Qurez','FAE Beauty','TNW The Natural Wash','Quench','Episoft','Sotrue','MCaffeine','Faces Canada','LANEIGE','Undry','Raaga Professional','O3+','Joy','Happier','Renee Cosmetics','Love Earth','Keya Seth','Moody','Thank You Farmer','Mountainor','Plix','Khadi Natural','Nirvasa','The Pink Foundry',"Cheryl's Cosmeceuticals",'Pilgrim','The Face Shop','Deconstruct','Ozone','ZM Zayn & Myza','Murad','Shiseido','Givenchy','Old School']
metadata_filter="Vertical:(face wash) AND Budget:[* to 500] AND Brand_name IN ['Lakme','Mama Earth']"
aatribute_list = ['Brand_Name','Product_Name','Vertical','Price','Ingredients','Texture','Chemical_Filter','Physical_Filter','Comedogenic_ingredients','SPF_UVB_Rating','SPF_UVA_Rating','Country_of_Origin','Fragrance_free','Oil_free','Brand_Type','Application_Area']

system_message_v2 = f"""
You are an expert in analyzing user queries related to skincare products. Your task is to perform two key functions:

**Part A: Convert Search Query to Marqo.ai Filter String**

- **Objective:** Transform the given search query into a Marqo.ai `filter_string` using the following steps:

    1. **Extract Field-Value Pairs:** Identify and extract all field-value pairs from the query. Example: `Vertical:(Sunscreen)`.
    2. **Combine Multiple Values:** If multiple values are specified for the same field (e.g., `Vertical:(Sunscreen)` , `Vertical:(Face Wash)`). Combine these into a single condition using the `OR` operator and enclose in parentheses. Example: `(Vertical:(Sunscreen) OR Vertical:(Face Wash))`.
    3. **Handle Spaces in Values:** Enclose values containing spaces within parentheses. Example: `SPF_UVB_Rating:(SPF 50)`.
    4. **Remove Unnecessary Quotes:** Eliminate single quotes around values. Example: `'Yes'` becomes `Yes`.
    5. **Handle Numeric Attributes:** Recognize that "Price" is the only numeric attribute. All other attributes should be treated as strings, including `SPF_UVB_Rating`, which requires specifying each possible value in a list for conditions like "SPF 50 or more."
    6. **Construct and Validate Filter String:** Rebuild the query into a valid Marqo.ai `filter_string`:
        - Maintain logical operators (AND, OR) as they are.
        - Ensure proper formatting of field-value pairs.
        - Ensure all multiple values are handled with the `OR` operator and enclose with `()`.
        - Validate that each condition is correctly formatted and logically combined.
    7. **Output:** Provide the validated `filter_string` suitable for use in Marqo.ai.

**Part B: Optimize Search Query for Vector Embedding**

- **Objective:** Rewrite the search query to make it concise, semantically rich, and optimal for vector embedding by following these steps:

    1. **Simplify Language:** Strip away unnecessary words, jargon, and complex phrases that don’t add semantic value. Focus on the essential keywords and concepts.

    2. **Focus on Core Concepts:** Identify the main ideas, keywords, or entities critical to conveying the search intent.

    3. **Remove Redundancies:** Eliminate redundant words or phrases that do not contribute additional meaning.

    4. **Preserve Context:** Ensure that the simplified query retains necessary context to avoid ambiguity or loss of meaning.

    5. **Rephrase for Clarity:** If the original query is unclear, rephrase it for better clarity and directness.

    6. **Output:** Provide the final, optimized query ready for vector embedding.

**Attributes to Extract:**

- **Attribute List (16 Total):** Strictly extract only these attributes from the user query. Do not infer or assume; extract based solely on the query keywords:

    1. **Brand_Name:** Match against f{brands}.
    2. **Product_Name:** Identify if specified.
    3. **Vertical:** Categorize into one of the predefined verticals (e.g., Face Wash, Sunscreen).
    4. **Price:** Extract if specified or implied (below 500 = cheap, above 1000 = expensive). This is the only numeric attribute.
    5. **Ingredients:** Match against INCIdecoder names.
    6. **Texture:** Identify texture type (e.g., Cream, Gel).
    7. **Chemical_filters:** "Yes", "No", or "Not specified".
    8. **Physical_filters:** "Yes", "No", or "Not specified".
    9. **Comedogenic_Ingredients:** "Yes", "No", or "Not specified".
    10. **SPF_UVB_Rating:** Extract SPF rating (e.g., SPF 50). Treat this as a string attribute and list out possible values for conditions like "SPF 50 or more." in this format "(SPF_UVB_Rating : (SPF 50) OR SPF_UVB_Rating : (SPF 60) OR SPF_UVB_Rating : (SPF 70))". Do Not create incorrect filter values like "SPF".
    11. **SPF_UVA_Rating:** Extract UVA rating (e.g., PA+++).
    12. **Country_of_origin:** Identify if specified.
    13. **Fragrance_free:** "Yes", "No", or "Not specified".
    14. **Oil_free:** "Yes", "No", or "Not specified".
    15. **Brand_Type:** Extract if mentioned (e.g., Drugstore, Luxury).
    16. **Application_Area:** Specify "Face" or "Body".

- **Combine Filters:** Combine filters using AND/OR as appropriate based on the query.

**Example Input:**
"I need a sunscreen and face wash from Neutrogena or Cetaphil that is fragrance-free and has an SPF of 50 or more."

**Example Output:**
- **Part A:** (Vertical :(Sunscreen) OR Vertical :(Face Wash)) AND (Brand_Name : (Neutrogena) OR Brand_Name : (Cetaphil)) AND Fragrance_free:Yes AND (SPF_UVB_Rating : (SPF 50) OR SPF_UVB_Rating : (SPF 60) OR SPF_UVB_Rating : (SPF 70))
- **Part B:** Fragrance-free Neutrogena or Cetaphil sunscreen and face wash with SPF 50+

**Instructions:**
Only extract the specified attributes that are present in the user query. If the query is ambiguous or lacks sufficient details, provide the best possible interpretation. Output only the results for Part A and Part B.


**Final Output:** Only share the required Part A and Part B. Exclude unwanted data in output.

"""

def chat_comepletion(user_query):
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_message_v2},
                {"role": "user", "content": user_query}],
            temperature=0,
            top_p=0, 
        stream=True,
    )
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    response = response.split('\n\n')
    return response

def get_parts_from_list(input_list):
    part_a = None
    part_b = None

    for item in input_list:
        item.replace('\n','')
        sent=item.strip()
        if sent.startswith("**Part A :**"):
            part_a = sent.split("**Part A :**")[1].strip()
        elif sent.startswith("**Part A:**"):
            part_a = sent.split("**Part A:**")[1].strip()
        elif sent.startswith("**Part B:**"):
            part_b = sent.split("**Part B:**")[1].strip()
        elif sent.startswith("**Part B :**"):
            part_b = sent.split("**Part B :**")[1].strip()
    return part_a, part_b

def search_query_product(search_query):
    filter = None
    query = None
    while query is None:
        filter,query = get_parts_from_list(chat_comepletion(search_query))

    results = mq.index("honestlysunscreen").search(
        q=query, filter_string=filter
    )
    product_name = [ hit['Product_Name'] for hit in results['hits']]
    product_score =  [ hit['_score'] for hit in results['hits']]
    Result = pd.DataFrame({
    'Product Name': product_name,
    'Product Score': product_score
    })
    return Result, filter, query

def main():
    st.title("Honestly Search Bar:")
    
    # Input search query
    query = st.text_input("Enter your search query:")
    
    # Search button
    if st.button("Search"):
        # Call the search function and display results
        st.header("Thinking....")
        results, filter, mod_user_query = search_query_product(query)
        st.header("Relevant Filters Extracted...")
        st.write(filter)
        st.write(" ")
        st.header("Updated Search Query...")
        st.write(mod_user_query)
        st.header("Recommended Products...")
        st.table(results)


if __name__ == "__main__":
    main()



