# Chatbot Inference
The response from the endpoint will contain either only the intent identified or the intent and an appropriate response to the message given, with the confidence score.

An example json response from the endpoint is given below:
```
{
    "prediction":
    [
        {
            "response": "How can I help you",
            "intent": "GREETING"
            "confidence": 1.
        }
    ]
}
```