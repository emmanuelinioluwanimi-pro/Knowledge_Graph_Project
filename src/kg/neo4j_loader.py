# neo4j_loader.py
from neo4j import GraphDatabase
import json

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def load_manual_data(self, data, product_code):
        """Load extracted data into Neo4j"""
        with self.driver.session() as session:
            # Create Manual
            session.run("""
                MERGE (m:Manual {productCode: $productCode})
                SET m.productName = $productName
            """, productCode=product_code, productName=data["manual"]["productName"])
            
            # Load Parts
            for part in data.get("parts", []):
                session.run("""
                    MATCH (m:Manual {productCode: $productCode})
                    MERGE (p:Part {partId: $partId})
                    SET p.name = $name, p.quantity = $quantity
                    MERGE (m)-[:CONTAINS_PART]->(p)
                """, productCode=product_code, **part)
            
            # Load Tools
            for tool in data.get("tools", []):
                session.run("""
                    MATCH (m:Manual {productCode: $productCode})
                    MERGE (t:Tool {toolName: $toolName, productCode: $productCode})
                    SET t.quantity = $quantity
                    MERGE (m)-[:REQUIRES_TOOL]->(t)
                """, productCode=product_code, **tool)
            
            # Load Steps + Relationships
            for step in data.get("steps", []):
                session.run("""
                    MATCH (m:Manual {productCode: $productCode})
                    MERGE (s:Step {productCode: $productCode, stepNumber: $stepNumber})
                    SET s.description = $description
                    MERGE (m)-[:HAS_STEP]->(s)
                """, productCode=product_code, **step)
                
                # Link parts used
                for part in step.get("partsUsed", []):
                    session.run("""
                        MATCH (s:Step {productCode: $productCode, stepNumber: $stepNumber})
                        MATCH (p:Part {partId: $partId})
                        MERGE (s)-[:USES_PART {quantity: $quantity}]->(p)
                    """, productCode=product_code, stepNumber=step["stepNumber"], **part)
                
                # Link tools used
                for tool_name in step.get("toolsUsed", []):
                    session.run("""
                        MATCH (s:Step {productCode: $productCode, stepNumber: $stepNumber})
                        MATCH (t:Tool {toolName: $toolName, productCode: $productCode})
                        MERGE (s)-[:USES_TOOL]->(t)
                    """, productCode=product_code, stepNumber=step["stepNumber"], toolName=tool_name)