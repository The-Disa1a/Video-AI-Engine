import os, requests, json
from vid_engine import context

def update_system_prompts():
    print("\n[📡] Fetching latest System Prompts from GitHub...", flush=True)

    urls = {
        "Video_and_Music_Sup.txt": "https://raw.githubusercontent.com/The-Disa1a/System_Contents/refs/heads/main/gemini/Video_and_Music_Sup",
        "gif_selector.txt": "https://raw.githubusercontent.com/The-Disa1a/System_Contents/refs/heads/main/gemini/gif_selector",
        "BGV_selector.txt": "https://raw.githubusercontent.com/The-Disa1a/System_Contents/refs/heads/main/gemini/BGV_selector"
    }

    prompts = {}
    for fname, url in urls.items():
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            text = r.text.strip()
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(text)
            prompts[fname] = text
            print(f"   [✅] Downloaded & Updated: {fname}", flush=True)
        except Exception as e:
            print(f"[⚠️] Failed to download {fname}: {e}", flush=True)
            if os.path.exists(fname):
                print(f"[♻️] Using existing local copy for {fname}", flush=True)
                with open(fname, 'r', encoding='utf-8') as f:
                    prompts[fname] = f.read().strip()
            else:
                raise Exception(f"Critical error: Could not fetch {fname} from GitHub and no local copy exists.")

    context.SYS_PROMPT_SUPERVISOR = prompts["Video_and_Music_Sup.txt"]
    context.SYS_PROMPT_GIF = prompts["gif_selector.txt"]
    context.SYS_PROMPT_BGV = prompts["BGV_selector.txt"]

def get_llm_keywords(sentences_list):
    try:
        from google import genai
        from google.genai import types

        numbered_text_lines =[f"{i+1}. Sentence: {s}" for i, s in enumerate(sentences_list)]
        numbered_text = "\n\n".join(numbered_text_lines)
        expected_len = len(sentences_list)

        system_instruction = context.SYS_PROMPT_SUPERVISOR

        schema_def = genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["global_bg_sound", "sentences"],
            properties = {
                "global_bg_sound": genai.types.Schema(type = genai.types.Type.STRING),
                "sentences": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        required =["scene_num", "bg_keywords", "popup_gifs", "wiki_images"],
                        properties = {
                            "scene_num": genai.types.Schema(type = genai.types.Type.STRING),
                            "bg_keywords": genai.types.Schema(type = genai.types.Type.STRING),
                            "popup_gifs": genai.types.Schema(
                                type = genai.types.Type.ARRAY,
                                items = genai.types.Schema(
                                    type = genai.types.Type.OBJECT,
                                    properties = {"keyword": genai.types.Schema(type = genai.types.Type.STRING), "search_query": genai.types.Schema(type = genai.types.Type.STRING)},
                                ),
                            ),
                            "wiki_images": genai.types.Schema(
                                type = genai.types.Type.ARRAY,
                                items = genai.types.Schema(
                                    type = genai.types.Type.OBJECT,
                                    properties = {"keyword": genai.types.Schema(type = genai.types.Type.STRING), "search": genai.types.Schema(type = genai.types.Type.STRING)},
                                )
                            )
                        },
                    ),
                ),
            },
        )

        for model_name in context.GEMINI_MODELS:
            print(f"[🧠] Connecting to Gemini API (Model: {model_name})...", flush=True)

            if "2.5" in model_name:
                t_cfg = types.ThinkingConfig(thinking_budget=-1)
            else:
                t_cfg = types.ThinkingConfig(thinking_level="HIGH")

            generate_content_config = types.GenerateContentConfig(
                thinking_config=t_cfg,
                response_mime_type="application/json",
                response_schema=schema_def,
                system_instruction=[types.Part.from_text(text=system_instruction)]
            )

            for attempt in range(2):
                if context.CURRENT_GEMINI_INDEX >= len(context.GEMINI_API_KEYS):
                    print("[❌] All Gemini API keys exhausted.", flush=True)
                    return None, None

                print(f"   -> Try {attempt+1}/2", flush=True)

                try:
                    current_api_key = context.GEMINI_API_KEYS[context.CURRENT_GEMINI_INDEX]
                    client = genai.Client(api_key=current_api_key)

                    response_stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=numbered_text)])],
                        config=generate_content_config,
                    )

                    print("   [📡] Receiving stream: ", end="", flush=True)
                    full_text = ""
                    for chunk in response_stream:
                        if chunk.text:
                            print(".", end="", flush=True)
                            full_text += chunk.text
                    print(" [Done!]", flush=True)

                    if full_text:
                        res_data = json.loads(full_text)
                        global_bg_sound = res_data.get("global_bg_sound", "cinematic ambient")

                        parsed_results =[]
                        for p_obj in res_data.get("sentences",[]):
                            bg_string = p_obj.get("bg_keywords", "nature environment")
                            parsed_results.append({
                                "bg_keywords": [bg_string],
                                "gifs": p_obj.get("popup_gifs",[]),
                                "wiki": p_obj.get("wiki_images",[])
                            })
                        if len(parsed_results) == expected_len:
                            print(f"\n[✅] Gemini ({model_name}) perfectly mapped {expected_len} scenes & assigned BGM: '{global_bg_sound}'\n", flush=True)
                            return parsed_results, global_bg_sound
                        else:
                            print(f"[⚠️] ERROR: Model returned {len(parsed_results)} scenes, expected {expected_len}.", flush=True)
                except Exception as e:
                    err_str = str(e)
                    print(f"\n[⚠️] ERROR from {model_name}: {err_str}", flush=True)
                    if "429" in err_str or "quota" in err_str.lower():
                        print(f"[⚠️] Gemini Key {context.CURRENT_GEMINI_INDEX} exhausted. Switching key...", flush=True)
                        context.CURRENT_GEMINI_INDEX += 1
                        continue 
                    if "503" in err_str:
                        print("[⚠️] 503 Server Error detected. Skipping this model.", flush=True)
                        break

        print("[❌] All Gemini models exhausted.", flush=True)
    except Exception as e:
        print(f"[❌] Critical Master Gemini Failure: {str(e)}", flush=True)
    return None, None
