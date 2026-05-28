How will you measure the performance of service you design?

There are two important metrics.

1. Response Time - How much time it takes for an response to be processed and served back to the user?

2. Throughput - How many requests can be processed in a given amount of time with a given hardware.

Unfortunately, both are kind of affecting one another. Like shown in this graph, trying to increase throughput on a given hardware will affect the response time due to queuing. Similarly, try to reduce the response time will will require you to scale the system. So the throughput reduces.

So, you should take an optimal middle ground and define your system's performance as such 1 million hits per second with p99 at 20 ms.

When you talk in appropriate terms, your chance of success increases.