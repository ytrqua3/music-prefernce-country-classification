import json
import io
import boto3

def lambda_handler(event, context):
    #load request content
    body = json.loads(event.get("body", "{}"))

    print("✅Successfully retreived user data from http request")
    print(body)
    ENDPOINT_NAME = "music-preference-v20"

    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",          
            Body=body
        )

    result = json.loads(response["Body"].read().decode('utf-8'))
    print(result)
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }

