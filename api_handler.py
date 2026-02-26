import json
import io
import boto3

def lambda_handler(event, context):
    #load request content
    body = json.loads(event.get("body", "{}"))

    print("✅Successfully retreived user data from http request")
    print(body)
    if ("top_artists" in body) and ("total_scrobbles" in body):
        if len(body["top_artists"]) != 50:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "top_artists must be of length 50"
                })
            }
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "Missing required fields"
            })
        }

    ENDPOINT_NAME = "music-preference-v25"
    body_bytes = json.dumps(body).encode('utf-8')
    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",          
            Body=body_bytes
        )

    result = json.loads(response["Body"].read().decode('utf-8'))
    print(result)
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }

