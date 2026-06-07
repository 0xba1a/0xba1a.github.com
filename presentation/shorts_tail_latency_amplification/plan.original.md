Imagine you have a database that's performance is 20ms at p99. So, only 1 in every 100 requests will take more than 20ms. You may be proud that you have designed a reliable data service.

But the backend engineer may complain the service is slow because of your database.

Because, your ecommerce business makes 10 data base queries for every user searches. And that request need to wait for all the 10 queries to complete before responding to the user. 10 query per user, so 100 query per 10 users, one among the query will be slower. So, one in 10 user will see a slower response.

Suddenly, an 1% bad latency is amplified to 10% of the users. It is called tail-latency amplification.
