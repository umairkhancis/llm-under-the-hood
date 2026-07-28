import chainlit

from inference_module.inference import generate_response

@chainlit.on_message
async def main(message: chainlit.Message):
    response = generate_response(message.content)

    await chainlit.Message(
        content=f"{response}",
    ).send()
