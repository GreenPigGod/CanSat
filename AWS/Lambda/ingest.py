import boto3, gzip, uuid, datetime, base64, os, json
s3 = boto3.client("s3")
BUCKET = os.environ.get("BUCKET", "pose-logs-2025")

def lambda_handler(event, context):
    try:
        body = event.get("body", "")
        if event.get("isBase64Encoded", False):
            body_bytes = base64.b64decode(body)
        else:
            # API GW の設定によってはプレーン文字列で来る
            body_bytes = body.encode("utf-8") if isinstance(body, str) else body

        now = datetime.datetime.utcnow()
        key = f"raw/android/{now:%Y/%m/%d/%H%M%S}-{uuid.uuid4().hex}.ndjson.gz"

        # 受信が未圧縮でも動くように判定（Android側はgzip送信でOK）
        if event.get("headers", {}).get("content-encoding", "").lower() == "gzip":
            payload = body_bytes
        else:
            payload = gzip.compress(body_bytes)

        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=payload,
            ContentEncoding="gzip",
            ContentType="application/x-ndjson",
        )
        print({"put": {"bucket": BUCKET, "key": key, "len": len(payload)}})
        return {"statusCode": 200, "body": "ok"}

    except Exception as e:
        # ここに出たメッセージが原因の決め手になります
        print({"error": str(e), "event_sample": str(event)[:512]})
        return {"statusCode": 500, "body": "error"}
