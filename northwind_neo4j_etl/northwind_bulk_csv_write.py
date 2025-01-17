import logging
import os
from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase
from retry import retry

CUSTOMERS_CSV_PATH = os.getenv("CUSTOMERS_CSV_PATH")
ORDERS_CSV_PATH = os.getenv("ORDERS_CSV_PATH")
PRODUCTS_CSV_PATH = os.getenv("PRODUCTS_CSV_PATH")
CATEGORIES_CSV_PATH = os.getenv("CATEGORIES_CSV_PATH")
SUPPLIERS_CSV_PATH = os.getenv("SUPPLIERS_CSV_PATH")
REVIEWS_CSV_PATH = os.getenv("REVIEWS_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

NODES = ["Customer", "Order", "Product", "Category", "Supplier", "Review"]

def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


@retry(tries=100, delay=10)
def load_northwind_graph_from_csv() -> None:
    """Load structured customers CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database="neo4j") as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)

    LOGGER.info("Loading customer nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{CUSTOMERS_CSV_PATH}' AS customers
        MERGE (c:Customer {{id: customers.customerID,
                            company_name: customers.companyName,
                            contact_name: customers.contactName,
                            contact_title: customers.contactTitle,
                            address: customers.address,
                            city: customers.city,
                            region: customers.region,
                            postal_code: customers.postalCode,
                            country: customers.country,
                            phone: customers.phone,
                            fax: customers.fax}});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading order nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS 
        FROM '{ORDERS_CSV_PATH}' AS orders
        MERGE (o:Order {{
            id: toInteger(orders.orderID),
            num_products: toInteger(orders.numProduct),
            unit_price: toInteger(orders.unitPrice),
            quantity: toInteger(orders.quantity),
            discount: toFloat(orders.discount),
            order_date: CASE 
            WHEN orders.orderDate IS NOT NULL AND orders.orderDate <> '' 
            THEN date({{year: toInteger(split(orders.orderDate, '/')[2]), 
                       month: toInteger(split(orders.orderDate, '/')[0]), 
                       day: toInteger(split(orders.orderDate, '/')[1])}})
            ELSE null 
            END,
        required_date: CASE 
            WHEN orders.requiredDate IS NOT NULL AND orders.requiredDate <> '' 
            THEN date({{year: toInteger(split(orders.requiredDate, '/')[2]), 
                       month: toInteger(split(orders.requiredDate, '/')[0]), 
                       day: toInteger(split(orders.requiredDate, '/')[1])}})
            ELSE null 
            END,
        shipped_date: CASE 
            WHEN orders.shippedDate IS NOT NULL AND orders.shippedDate <> '' 
            THEN date({{year: toInteger(split(orders.shippedDate, '/')[2]), 
                       month: toInteger(split(orders.shippedDate, '/')[0]), 
                       day: toInteger(split(orders.shippedDate, '/')[1])}})
            ELSE null 
            END,
            ship_via: orders.shipVia,
            freight: orders.freight,
            ship_name: orders.shipName,
            ship_city: orders.shipCity,
            ship_postal_code: orders.shipPostalCode,
            ship_country: orders.shipCountry
    }});
    """
        _ = session.run(query, {})
    
    LOGGER.info("Loading product nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PRODUCTS_CSV_PATH}' AS products
        MERGE (p:Product {{id: toInteger(products.productID),
        product_name: products.productName,
        quantity_per_unit: products.quantityPerUnit,
        unit_price: toFloat(products.unitPrice),
        units_in_stock: toInteger(products.unitsInStock),
        units_on_order: toInteger(products.unitsOnOrder),
        reorder_level: toInteger(products.reorderLevel),
        discontinued: toInteger(products.discontinued)}});
        """
        _ = session.run(query, {})
    
    LOGGER.info("Loading supplier nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{SUPPLIERS_CSV_PATH}' AS suppliers
        MERGE (s:Supplier {{id: toInteger(suppliers.supplierID),
        company_name: suppliers.companyName,
        contact_name: suppliers.contactName,
        contact_title: suppliers.contactTitle,
        supplier_address: suppliers.address,
        supplier_city: suppliers.city,
        supplier_region: suppliers.region,
        supplier_postal_code: suppliers.postalCode,
        supplier_country: suppliers.country,
        supplier_phone: suppliers.phone,
        supplier_fax: suppliers.fax}});
        """
        _ = session.run(query, {})
        
    LOGGER.info("Loading category nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{CATEGORIES_CSV_PATH}' AS categories
        MERGE (c:Category {{id: toInteger(categories.categoryID),
        category_name: categories.categoryName,
        category_description: categories.description}});
        """
        _ = session.run(query, {})
        
    LOGGER.info("Loading review nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{REVIEWS_CSV_PATH}' AS reviews
        MERGE (r:Review {{id: toInteger(reviews.reviewID),
        text: reviews.reviews}});
        """
        _ = session.run(query, {})
    

    LOGGER.info("Loading 'PURCHASED' relationships")
    # specify the file having both the keys
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{ORDERS_CSV_PATH}' AS row
            MATCH (c:Customer {{id: toString(row.customerID)}})
            MATCH (o:Order {{id: toInteger(row.orderID)}})
            MERGE (c)-[purchased:PURCHASED]->(o)
        """
        _ = session.run(query, {})
    
    LOGGER.info("Loading 'ORDERS' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{ORDERS_CSV_PATH}' AS row
            MATCH (o:Order {{id: toInteger(row.orderID)}})
            MATCH (p:Product {{id: toInteger(row.productID)}})
            MERGE (o)-[orders:ORDERS]->(p)
        """
        _ = session.run(query, {})
        
    LOGGER.info("Loading 'SUPPLIES' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{PRODUCTS_CSV_PATH}' AS row
            MATCH (s:Supplier {{id: toInteger(row.supplierID)}})
            MATCH (p:Product {{id: toInteger(row.productID)}})
            MERGE (s)-[supplies:SUPPLIES]->(p)
        """
        _ = session.run(query, {})
        
    LOGGER.info("Loading 'PART_OF' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{PRODUCTS_CSV_PATH}' AS row
            MATCH (c:Category {{id: toInteger(row.categoryID)}})
            MATCH (p:Product {{id: toInteger(row.productID)}})
            MERGE (p)-[part_of:PART_OF]->(c)
        """
        _ = session.run(query, {})
        
    LOGGER.info("Loading 'WRITES' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{REVIEWS_CSV_PATH}' AS row
            MATCH (o:Order {{id: toInteger(row.orderID)}})
            MATCH (r:Review {{id: toInteger(row.reviewID)}})
            MERGE (o)-[writes:WRITES]->(r)
        """
        _ = session.run(query, {})    


if __name__ == "__main__":
    load_northwind_graph_from_csv()
