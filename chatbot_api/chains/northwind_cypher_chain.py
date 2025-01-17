import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

NORTHWIND_QA_MODEL = os.getenv("NORTHWIND_QA_MODEL")
NORTHWIND_CYPHER_MODEL = os.getenv("NORTHWIND_CYPHER_MODEL")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH o.order_date as orderDate, p.unitPrice as unitPrice)
If you need to divide numbers, make sure to
filter the denominator to be non zero.

Examples:
# Find total quantity per customer in the "Produce" category in year 2012?
MATCH (cust:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)-[:PART_OF]->(c:Category {category_name:"Produce"})
WHERE o.order_date >= date('2012-01-01') AND o.order_date < date('2013-01-01')
RETURN DISTINCT cust.contact_name as CustomerName, SUM(o.quantity) AS TotalProductsPurchased

# Which customer(s) has ordered orders with more than 5 products in it?
MATCH (c:Customer)-[:PURCHASED]->(o:Order)
WHERE o.num_products > 5
RETURN c.company_name, o.id;

# What are the product categories provided by each supplier?
MATCH (s:Supplier)-[:SUPPLIES]->(p:Product)-[:PART_OF]->(c:Category)
RETURN s.company_name as Company, collect(distinct c.category_name) as Categories;

# How many customers in Germany have written reviews?
MATCH (c:Customer)-[:PURCHASED]->(o:Order)-[:WRITES]->(r:Review)
WHERE c.country = 'Germany'
RETURN COUNT(DISTINCT c) AS customers_with_reviews;

# Who are the suppliers supplying products in "Produce" category?
MATCH (s:Supplier)-[:SUPPLIES]->(p:Product)-[:PART_OF]->(c:Category)
WHERE c.name = 'Produce'
RETURN DISTINCT s.company_name as Supplier, collect(p.name) as Products
ORDER BY s.company_name

# What is the net sales revenue in year 2012?
MATCH (o:Order) 
WHERE o.order_date >= date('2012-01-01') AND o.order_date < date('2013-01-01')
RETURN 'Total Sales is: $' + toInteger(SUM(o.quantity*((1-o.discount)*o.unit_price)))

# Which country had the largest percent increase in number of orders from 2012 to 2013?
MATCH (c:Customer)-[:PURCHASED]->(o:Order)
WHERE o.order_date >= date('2012-01-01') AND o.order_date <= date('2013-12-31')
WITH c.country AS country, COUNT(o) AS order_count,
SUM(CASE WHEN o.order_date >= date('2012-01-01') AND o.order_date < date('2013-01-01') THEN 1 ELSE 0 END) AS count_2012,
SUM(CASE WHEN o.order_date >= date('2013-01-01') AND o.order_date < date('2014-01-01')
THEN 1 ELSE 0 END) AS count_2013
WITH country, order_count, count_2012, count_2013,
     (toFloat(count_2013) - toFloat(count_2012)) / toFloat(count_2012) * 100
     AS percent_increase
RETURN country, percent_increase
ORDER BY percent_increase DESC
LIMIT 1;

String category values:
Category names are one of: 'Beverages', 'Condiments', 'Confections', 'Dairy Products', 'Grains/Cereals', 'Meat/Poultry', 'Produce', 'Seafood'
Product names are one of: 'Chai', 'Chang', 'Aniseed Syrup', 'Chef Anton's Cajun Seasoning', 'Chef Anton's Gumbo Mix', 'Grandma's Boysenberry Spread', 
'Uncle Bob's Organic Dried Pears', 'Northwoods Cranberry Sauce', 'Mishi Kobe Niku', 'Ikura', 'Queso Cabrales', 'Queso Manchego La Pastora', 'Konbu', 'Tofu', 'Genen Shouyu', 
'Pavlova', 'Alice Mutton', 'Carnarvon Tigers', 'Teatime Chocolate Biscuits', 'Sir Rodney's Marmalade', 'Sir Rodney's Scones', 'Gustaf's KnÃ¤ckebrÃ¶d', 'TunnbrÃ¶d', 
'GuaranÃ¡ FantÃ¡stica', 'NuNuCa NuÃŸ-Nougat-Creme', 'GumbÃ¤r GummibÃ¤rchen', 'Schoggi Schokolade', 'RÃ¶ssle Sauerkraut', 'ThÃ¼ringer Rostbratwurst', 'Nord-Ost Matjeshering', 
'Gorgonzola Telino', 'Mascarpone Fabioli', 'Geitost', 'Sasquatch Ale', 'Steeleye Stout', 'Inlagd Sill', 'Gravad lax', 'CÃ´te de Blaye', 'Chartreuse verte', 'Boston Crab Meat', 
'Jack's New England Clam Chowder', 'Singaporean Hokkien Fried Mee', 'Ipoh Coffee', 'Gula Malacca', 'Rogede sild', 'Spegesild', 'Zaanse koeken', 'Chocolade', 'Maxilaku', 
'Valkoinen suklaa', 'Manjimup Dried Apples', 'Filo Mix', 'Perth Pasties', 'TourtiÃ¨re', 'PÃ¢tÃ© chinois', 'Gnocchi di nonna Alice', 'Ravioli Angelo', 'Escargots de Bourgogne', 
'Raclette Courdavault', 'Camembert Pierrot', 'Sirop d'Ã©rable', 'Tarte au sucre', 'Vegie-spread', 'Wimmers gute SemmelknÃ¶del', 'Louisiana Fiery Hot Pepper Sauce', 
'Louisiana Hot Spiced Okra', 'Laughing Lumberjack Lager', 'Scottish Longbreads', 'Gudbrandsdalsost', 'Outback Lager', 'Flotemysost', 'Mozzarella di Giovanni', 'RÃ¶d Kaviar', 
'Longlife Tofu', 'RhÃ¶nbrÃ¤u Klosterbier', 'LakkalikÃ¶Ã¶ri', 'Original Frankfurter grÃ¼ne SoÃŸe'
Suppliers are one of: 'Exotic Liquids', 'New Orleans Cajun Delights', 'Grandma Kelly's Homestead', 'Tokyo Traders', 'Cooperativa de Quesos 'Las Cabras'', 'Mayumi's', 
'Pavlova', 'Specialty Biscuits', 'PB KnÃ¤ckebrÃ¶d AB', 'Refrescos Americanas LTDA', 'Heli SÃ¼ÃŸwaren GmbH & Co. KG', 'Plutzer LebensmittelgroÃŸmÃ¤rkte AG', 
'Nord-Ost-Fisch Handelsgesellschaft mbH', 'Formaggi Fortini s.r.l.', 'Norske Meierier', 'Bigfoot Breweries', 'Svensk SjÃ¶fÃ¶da AB', 'Aux joyeux ecclÃ©siastiques', 
'New England Seafood Cannery', 'Leka Trading', 'Lyngbysild', 'Zaanse Snoepfabriek', 'Karkki Oy', 'G'day', 'Ma Maison', 'Pasta Buttini s.r.l.', 'Escargots Nouveaux', 
'Gai pÃ¢turage', 'ForÃªts d'Ã©rables'


Make sure to use IS NULL or IS NOT NULL when analyzing missing properties.
Never return embedding properties in your queries. You must never include the
statement "GROUP BY" in your query. Make sure to alias all statements that
follow as with statement (e.g. WITH v as visit, c.billing_amount as
billing_amount)
If you need to divide numbers, make sure to filter the denominator to be non
zero.

The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a users natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

When names are provided in the query results, such as customer names,
beware  of any names that have commas or other punctuation in them.
For instance, 'Jones, Brown and Murray' is a single customer or company name,
not multiple customers. Make sure you return any list of names in
a way that isn't ambiguous and allows someone to tell what the full
names are.

Never say you don't have the right information if there is data in
the query results. Make sure to show all the relevant query results
if you're asked.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

northwind_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=NORTHWIND_CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=NORTHWIND_QA_MODEL, temperature=0),
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)
