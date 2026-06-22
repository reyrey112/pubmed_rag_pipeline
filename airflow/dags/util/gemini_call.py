from google.genai import types, errors
import time

def gemini_call(client, model, prompt,max_retries, safety_settings = None):
    retry_count = 0
    while retry_count < max_retries:
        try:
            print("Calling Gemini")
            response: types.GenerateContentResponse
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    response_mime_type="application/json",
                    safety_settings=safety_settings,
                    # system_instruction="",
                ),
            )

            return response
        
        except errors.APIError as e:
            print("API Error")
            if e.code in [503, 429]:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return prompt #edit
                
                print(f"Model busy (Status {e.code}). Retrying in 2.5 seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                time.sleep(2.5)
            
            else:
                # for permanent errors
                raise e
        
        except Exception as e:
            # non-api errors
            print(f"Unexpected error: {e}")
            raise e
    print("returning prompt")
    return prompt