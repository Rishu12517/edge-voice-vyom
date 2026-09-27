from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.get("/")
async def root():
    return {
        "status": "running",
        "project": "Edge Voice Activator",
        "service": "Backend"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    print("================================")
    print("ESP32 WebSocket CONNECTED")
    print("================================")

    try:
        while True:
            message = await websocket.receive_text()

            print("ESP32 says:", message)

            await websocket.send_text(
                "Message received by FastAPI"
            )

    except WebSocketDisconnect:
        print("================================")
        print("ESP32 WebSocket DISCONNECTED")
        print("================================")

    except Exception as e:
        print("WebSocket error:", e)