-- Customer Churn Analysis
WITH customer_segments AS (
    SELECT 
        customer_id,
        CASE 
            WHEN days_since_last_purchase < 30 THEN 'Active'
            WHEN days_since_last_purchase BETWEEN 30 AND 90 THEN 'At Risk'
            ELSE 'Churned'
        END AS churn_status,
        total_purchases,
        avg_order_value
    FROM customers
),
churn_prediction AS (
    SELECT 
        churn_status,
        COUNT(*) as customer_count,
        AVG(total_purchases) as avg_total_purchases,
        AVG(avg_order_value) as avg_customer_value,
        {{nql.predict_churn(churn_status, 'customer_segments')}} as churn_probability
    FROM customer_segments
    GROUP BY churn_status
)
SELECT 
    churn_status, 
    customer_count, 
    avg_total_purchases, 
    avg_customer_value,
    churn_probability
FROM churn_prediction;