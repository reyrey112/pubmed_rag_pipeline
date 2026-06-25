from google.genai import types, errors
import time

MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
MAX_OUTPUT_TOKENS = 1000

def gemini_call(
    client,
    prompt,
    response_mime_type=None,
    max_retries=MAX_RETRIES,
    model=MODEL,
    safety_settings=None,
    max_output_tokens=MAX_OUTPUT_TOKENS,
):
    start_time = 1
    retry_count = 0
    while retry_count < max_retries:
        try:
            print("Calling Gemini")
            response: types.GenerateContentResponse
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    response_mime_type=response_mime_type,
                    safety_settings=safety_settings,
                    # system_instruction="",
                ),
            )
    
            print("Returning response ")
            return response

        except errors.APIError as e:
            print("API Error")
            if e.code in [503, 429]:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return prompt  # edit

                print(
                    f"Model busy (Status {e.code}). Retrying in {start_time} seconds... (Attempt {retry_count}/{MAX_RETRIES})"
                )
                time.sleep(start_time)
                start_time *= 2

            else:
                # for permanent errors
                raise e

        except Exception as e:
            # non-api errors
            print(f"Unexpected error: {e}")
            raise e
    print("returning prompt")
    return prompt
