# Structured JSON Extraction via VLMM

Once you have the github repo cloned and modal set up simply run to try the application

```bash
modal serve main.py
```

To test to go `<URL>/docs` or make a request to something that looks like:

```bash
curl -X 'POST' \
  '<URL>' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
   "system":"Extract Users from `Interface Users { users: Array<{name: string, age:number}>}`",
   "data":[
      "James, 33, and Isabella, 23, are among the users with Benjamin, who is 34, Mia, 30, and Ethan, 28.",
      "Evelyn, 25, and Jacob, 29, are part of the records, along with Abigail, 27, Liam, 32, and Harper, 26."
   ]
}'
```

This request will produce results that roughtly look like:

```python
{
   "data":[
        {
        "users":[
            {"name":"James", "age":33},
            {"name":"Isabella", "age":23},
            {"name":"Benjamin", "age":34},
            {"name":"Mia", "age":30},
            {"name":"Ethan", "age":28}]}, 
        {
         "users":[
            {"name":"Evelyn", "age":25},
            {"name":"Jacob", "age":29},
            {"name":"Abigail", "age":27},
            {"name":"Liam", "age":32},
            {"name":"Harper", "age":26}] 
        }
   ],
   "num_tokens":261
}
```