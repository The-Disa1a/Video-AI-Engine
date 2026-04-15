import os, json, re, requests, subprocess, urllib.request, urllib.parse, glob
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from vid_engine import context

def get_youtube_gameplay(game_name):
    search_query = f"{game_name} gameplay no commentary creative commons"
    print(f"\n[📡] Searching YouTube for global gameplay hook: '{search_query}'...", flush=True)

    cmd =["yt-dlp", f"ytsearch10:{search_query}", "--dump-json", "--no-playlist", "--flat-playlist"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    all_videos =[]
    for line in result.stdout.strip().split("\n"):
        if not line: continue
        try:
            data = json.loads(line)
            vid_id = data.get("id")
            title = data.get("title")
            dur = data.get("duration", 0)
            if vid_id and title and dur > 60:
                all_videos.append({"id": vid_id, "title": title, "duration": dur})
        except: pass

    if not all_videos:
        print(f"[⚠️] No YouTube gameplay found for {game_name}.", flush=True)
        return None

    ranked_id = all_videos[0]['id']
    ranked_dur = all_videos[0]['duration']
    
    try:
        from google import genai
        from google.genai import types

        system_instruction = "You are a YouTube Gaming aesthetic evaluator. Pick the best high-quality gameplay video to use as a background hook. Avoid weird mods or tutorials. Return a JSON with the key 'id'."
        
        schema_def = genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required =["id"],
            properties = {"id": genai.types.Schema(type = genai.types.Type.STRING)},
        )

        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            response_mime_type="application/json",
            response_schema=schema_def,
            system_instruction=[types.Part.from_text(text=system_instruction)]
        )

        prompt = f"Target Game: {game_name}\nAvailable Videos:\n"
        for v in all_videos: prompt += f"- ID: {v['id']} | Title: {v['title']}\n"

        for m in context.GEMMA_MODELS:
            success = False
            while context.CURRENT_GEMINI_INDEX < len(context.GEMINI_API_KEYS):
                try:
                    client = genai.Client(api_key=context.GEMINI_API_KEYS[context.CURRENT_GEMINI_INDEX])
                    resp = client.models.generate_content(
                        model=m,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                        config=cfg
                    )
                    
                    clean_text = re.sub(r'^```(?:json)?\s*', '', resp.text.strip())
                    clean_text = re.sub(r'\s*```$', '', clean_text)
                    parsed_json = json.loads(clean_text)

                    if "id" in parsed_json:
                        selected_id = parsed_json["id"]
                        valid_match = next((v for v in all_videos if v['id'] == selected_id), None)
                        if valid_match:
                            ranked_id = valid_match['id']
                            ranked_dur = valid_match['duration']
                            print(f"[🤖 Gemma Choosed]: {valid_match['title']}", flush=True)
                            success = True
                            break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower(): context.CURRENT_GEMINI_INDEX += 1
                    else: break
            if success: break
    except: pass

    start_time = min(int(ranked_dur * 0.2), max(0, ranked_dur - 180))
    end_time = start_time + 180
    
    # Check if we already have it from a previous run
    existing = glob.glob(f"yt_bg_{ranked_id}.*")
    if existing:
        print(f"[✅] YouTube hook already exists: {existing[0]}", flush=True)
        return existing[0]

    print(f"   📦 Slicing YouTube Video '{ranked_id}' (Extracting {start_time}s to {end_time}s)...", flush=True)
    out_tmpl = f"yt_bg_{ranked_id}.%(ext)s"
    
    dl_cmd =[
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]/bestvideo[ext=mp4]/best",
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-overwrites",
        "-o", out_tmpl,
        f"https://www.youtube.com/watch?v={ranked_id}"
    ]
    
    result = subprocess.run(dl_cmd, capture_output=True, text=True)
    
    # Dynamically find the file regardless of extension
    downloaded = glob.glob(f"yt_bg_{ranked_id}.*")
    if downloaded:
        print(f"[✅] YouTube hook successfully sliced and downloaded! ({downloaded[0]})", flush=True)
        return downloaded[0]
    else:
        print(f"   [⚠️] yt-dlp slice failed. Error:\n{result.stderr}", flush=True)
        
    return None

def fetch_and_choose_bgm(mood_phrase):
    search_query = f"{mood_phrase} background music audio library no copyright"
    print(f"[📡] Searching SoundCloud via yt-dlp for: '{search_query}'...", flush=True)

    bgm_filename = "downloaded_bgm.mp3"
    if os.path.exists(bgm_filename): os.remove(bgm_filename)

    try:
        cmd =[
            "yt-dlp", f"scsearch1:{search_query}",
            "--extract-audio", "--audio-format", "mp3",
            "--output", bgm_filename
        ]
        print("      -> Fetching and downloading track from SoundCloud...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if os.path.exists(bgm_filename):
            print("[✅] Successfully downloaded pure BGM track!", flush=True)
            return bgm_filename
        else:
            print(f"      [⚠️] yt-dlp download failed to create file. Error log: {result.stderr}", flush=True)
    except Exception as e:
        print(f"      [⚠️] yt-dlp execution failed: {e}", flush=True)
    return None

def scrape_wikipedia_image(search_title):
    try:
        clean_title = search_title.strip().replace(" ", "_")
        q = urllib.parse.quote(clean_title)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
        data = json.loads(res)

        img_url = None
        if "thumbnail" in data: img_url = data["thumbnail"]["source"]
        elif "originalimage" in data: img_url = data["originalimage"]["source"]

        if img_url:
            fname = f"wiki_{clean_title}.jpg"
            if not os.path.exists(fname):
                with open(fname, "wb") as f:
                    f.write(urllib.request.urlopen(urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})).read())
            return fname
    except: pass
    return None

def get_giphy_gif(search_query, sentence_context):
    try:
        from google import genai
        from google.genai import types

        clean_query = urllib.parse.quote(search_query.strip())
        gif_list =[]

        while context.CURRENT_GIPHY_INDEX < len(context.GIPHY_API_KEYS):
            api_key = context.GIPHY_API_KEYS[context.CURRENT_GIPHY_INDEX]
            url = f"https://api.giphy.com/v1/gifs/search?api_key={api_key}&q={clean_query}&limit=15&rating=pg"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                res = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
                data = json.loads(res)
                if data.get("data"):
                    gif_list = [{"id": g["id"], "title": g["title"], "url": g["images"]["downsized"]["url"]} for g in data["data"]]
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    print(f"[⚠️] Giphy Key {context.CURRENT_GIPHY_INDEX} exhausted. Switching key...", flush=True)
                    context.CURRENT_GIPHY_INDEX += 1
                else:
                    print(f"[⚠️] Giphy fetch failed for '{search_query}': {e}", flush=True)
                    break

        if not gif_list: return None

        if context.ADV_OUTPUT:
            print(f"\n      --- 🔎 GIPHY SEARCH RESULTS FOR '{search_query}' ---", flush=True)
            for g in gif_list: print(f"         - ID: {g['id']} | Title: {g['title']}", flush=True)
            print("      --------------------------------------------------", flush=True)

        system_instruction = context.SYS_PROMPT_GIF
        schema_def = genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["matches"],
            properties={
                "matches": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["id"],
                        properties={"id": genai.types.Schema(type=genai.types.Type.STRING)},
                    ),
                ),
            },
        )

        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            response_mime_type="application/json",
            response_schema=schema_def,
            system_instruction=[types.Part.from_text(text=system_instruction)]
        )

        prompt = f"Video Sentence: {sentence_context}\nAvailable GIFs from Giphy:\n"
        for g in gif_list: prompt += f"- ID: {g['id']} | Title: {g['title']}\n"

        ranked_ids =[]
        if context.ADV_OUTPUT: print(f"[🧠] Gemma is evaluating and ranking {len(gif_list)} Giphy results...", flush=True)

        for m in context.GEMMA_MODELS:
            success = False
            while context.CURRENT_GEMINI_INDEX < len(context.GEMINI_API_KEYS):
                try:
                    client = genai.Client(api_key=context.GEMINI_API_KEYS[context.CURRENT_GEMINI_INDEX])
                    resp_stream = client.models.generate_content_stream(
                        model=m,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                        config=cfg
                    )

                    print("[📡] Stream: ", end="", flush=True)
                    full_text = ""
                    for chunk in resp_stream:
                        if chunk.text:
                            print(".", end="", flush=True)
                            full_text += chunk.text
                    print("[Done!]", flush=True)

                    clean_text = re.sub(r'^```(?:json)?\s*', '', full_text.strip())
                    clean_text = re.sub(r'\s*```$', '', clean_text)
                    parsed_json = json.loads(clean_text)

                    if "matches" in parsed_json and parsed_json["matches"]:
                        ranked_ids = [item["id"] for item in parsed_json["matches"] if "id" in item]
                        if ranked_ids:
                            display_list = [f"{rid}:{next((g['title'] for g in gif_list if g['id'] == rid), 'Unknown')}" for rid in ranked_ids]
                            print(f"[🤖 Gif's Gemma Choosed]: {json.dumps(display_list)}", flush=True)
                            success = True
                            break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower():
                        context.CURRENT_GEMINI_INDEX += 1
                    else: break 
            if success: break

        for g in gif_list:
            if g['id'] not in ranked_ids: ranked_ids.append(g['id'])

        for rid in ranked_ids:
            gif_obj = next((g for g in gif_list if g["id"] == rid), None)
            if not gif_obj: continue

            fname = f"giphy_{rid}.gif"
            if not os.path.exists(fname):
                with open(fname, "wb") as f:
                    f.write(urllib.request.urlopen(urllib.request.Request(gif_obj['url'], headers={'User-Agent': 'Mozilla/5.0'})).read())

            try:
                img = Image.open(fname)
                frames_to_check =[]

                if getattr(img, "is_animated", False):
                    total_frames = img.n_frames
                    step = max(1, total_frames // 5)
                    for i in range(0, total_frames, step):
                        if len(frames_to_check) < 5:
                            img.seek(i)
                            frames_to_check.append(img.convert('RGB'))
                else:
                    frames_to_check.append(img.convert('RGB'))
                img.close()

                rejected = False
                for f_idx, frame_img in enumerate(frames_to_check):
                    text_extracted = pytesseract.image_to_string(frame_img).strip()
                    word_count = len([w for w in text_extracted.split() if w.isalnum()])

                    if word_count > 3:
                        if context.ADV_OUTPUT: print(f"      [🔍] OCR rejected text-heavy GIF '{gif_obj['title']}'. Trying next...", flush=True)
                        rejected = True
                        break

                if rejected:
                    os.remove(fname)
                    continue
                else:
                    if context.ADV_OUTPUT: print(f"      [✅] GIF '{gif_obj['title']}' passed multi-frame OCR checks!", flush=True)
                    return fname, gif_obj['title']

            except Exception as e:
                print(f"      [⚠️] OCR failure on '{gif_obj['title']}': {e}", flush=True)
                return fname, gif_obj['title']

    except Exception as e: pass
    return None

def make_popup(path, is_wiki=False, card_label=""):
    try: img = Image.open(path)
    except: return None

    frames =[]
    is_animated = getattr(img, "is_animated", False)

    dur_ms = img.info.get('duration', 33)
    if dur_ms == 0: dur_ms = 33
    native_fps = 1000.0 / dur_ms

    try:
        while True:
            if is_wiki:
                rgba_img = img.convert("RGBA")
                w, h = rgba_img.size

                base_max = 400
                ratio = base_max / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                processed = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                pad_x, pad_y = 15, 15
                try: font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 28)
                except: font = ImageFont.load_default()

                card_label_clean = card_label.upper()

                if card_label_clean:
                    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
                    tw = dummy_draw.textlength(card_label_clean, font=font)
                    text_h = 45
                else:
                    tw, text_h = 0, 0

                card_w = max(new_w + pad_x * 2, int(tw) + pad_x * 2) if card_label_clean else new_w + pad_x * 2
                card_h = new_h + pad_y * 2 + text_h

                canvas = Image.new("RGBA", (card_w, card_h), (0,0,0,0))
                card_draw = ImageDraw.Draw(canvas)
                card_draw.rounded_rectangle((0, 0, card_w, card_h), radius=15, fill=(255, 255, 255, 255))

                img_x = (card_w - new_w) // 2
                canvas.paste(processed, (img_x, pad_y), processed if processed.mode == "RGBA" else None)

                if card_label_clean:
                    if tw > card_w - 20:
                        card_label_clean = card_label_clean[:15] + "..."
                        tw = card_draw.textlength(card_label_clean, font=font)
                    tx = (card_w - tw) // 2
                    ty = card_h - 40
                    card_draw.text((int(tx), int(ty)), card_label_clean, font=font, fill=(0,0,0, 255))

                frames.append(canvas)
            else:
                rgba_img = img.convert("RGBA")
                w, h = rgba_img.size
                base_max = 400
                ratio = base_max / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                resized = rgba_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                frames.append(resized)

            if not is_animated: break
            img.seek(img.tell() + 1)
    except EOFError: pass

    if len(frames) == 1: return {"frames": frames, "fps": 1.0, "type": "static"}
    return {"frames": frames, "fps": native_fps, "type": "animated"}

def get_background_videos(keywords, target_duration, prefix_idx, sentence_context=""):
    if not keywords: keywords = ["nature"]
    files, cur_dur =[], 0.0
    all_videos =[]

    for kw in keywords:
        q = requests.utils.quote(kw)
        while context.CURRENT_PIXABAY_INDEX < len(context.PIXABAY_API_KEYS):
            api_key = context.PIXABAY_API_KEYS[context.CURRENT_PIXABAY_INDEX]
            try:
                r = requests.get(f"https://pixabay.com/api/videos/?key={api_key}&q={q}&per_page=10", timeout=4)
                if r.status_code == 200:
                    if r.json().get("hits"):
                        for hit in r.json()["hits"]:
                            vid_url = hit["videos"].get("medium", hit["videos"].get("large"))["url"]
                            vid_id = f"pix_{hit['id']}"
                            all_videos.append({"id": vid_id, "url": vid_url, "dur": hit.get("duration", 10.0), "source": "Pixabay", "desc": hit.get("tags", kw)})
                    break
                elif r.status_code == 429: context.CURRENT_PIXABAY_INDEX += 1
                else: break
            except: break

        while context.CURRENT_PEXELS_INDEX < len(context.PEXELS_API_KEYS):
            api_key = context.PEXELS_API_KEYS[context.CURRENT_PEXELS_INDEX]
            try:
                r = requests.get(f"https://api.pexels.com/videos/search?query={q}&per_page=10&orientation=landscape", headers={"Authorization": api_key}, timeout=4)
                if r.status_code == 200:
                    if r.json().get("videos"):
                        for hit in r.json()["videos"]:
                            hd =[f for f in hit.get("video_files", []) if f['quality']=='hd']
                            vid_url = hd[0]['link'] if hd else hit["video_files"][0]['link']
                            vid_id = f"pex_{hit['id']}"
                            desc_slug = hit.get("url", "").split("/")[-2].replace("-", " ") if "url" in hit else kw
                            all_videos.append({"id": vid_id, "url": vid_url, "dur": float(hit.get("duration", 10.0)), "source": "Pexels", "desc": desc_slug})
                    break
                elif r.status_code == 429: context.CURRENT_PEXELS_INDEX += 1
                else: break
            except: break

    if not all_videos:
        if "nature" not in keywords: return get_background_videos(["nature"], target_duration, prefix_idx, sentence_context)
        raise Exception("No Background Videos found.")

    ranked_ids =[]
    if sentence_context and len(all_videos) > 1:
        try:
            from google import genai
            from google.genai import types

            system_instruction = context.SYS_PROMPT_BGV
            schema_def = genai.types.Schema(
                type = genai.types.Type.OBJECT,
                required =["matches"],
                properties = {
                    "matches": genai.types.Schema(
                        type = genai.types.Type.ARRAY,
                        items = genai.types.Schema(
                            type = genai.types.Type.OBJECT,
                            required = ["id"],
                            properties = {"id": genai.types.Schema(type = genai.types.Type.STRING)},
                        ),
                    ),
                },
            )

            cfg = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
                response_mime_type="application/json",
                response_schema=schema_def,
                system_instruction=[types.Part.from_text(text=system_instruction)]
            )

            prompt = f"Video Sentence: {sentence_context}\nAvailable Background Videos:\n"
            for v in all_videos: prompt += f"- ID: {v['id']} | Desc: {v['desc']} | Source: {v['source']}\n"

            for m in context.GEMMA_MODELS:
                success = False
                while context.CURRENT_GEMINI_INDEX < len(context.GEMINI_API_KEYS):
                    try:
                        client = genai.Client(api_key=context.GEMINI_API_KEYS[context.CURRENT_GEMINI_INDEX])
                        resp_stream = client.models.generate_content_stream(
                            model=m,
                            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                            config=cfg
                        )

                        full_text = "".join([chunk.text for chunk in resp_stream if chunk.text])
                        clean_text = re.sub(r'^```(?:json)?\s*', '', full_text.strip())
                        clean_text = re.sub(r'\s*```$', '', clean_text)
                        parsed_json = json.loads(clean_text)

                        if "matches" in parsed_json and parsed_json["matches"]:
                            ranked_ids = [item["id"] for item in parsed_json["matches"] if "id" in item]
                            if ranked_ids:
                                success = True
                                break
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "quota" in err_str.lower(): context.CURRENT_GEMINI_INDEX += 1
                        else: break
                if success: break
        except: pass

    ranked_videos =[]
    for rid in ranked_ids:
        for v in all_videos:
            if v['id'] == rid and v not in ranked_videos: ranked_videos.append(v)
    for v in all_videos:
        if v not in ranked_videos: ranked_videos.append(v)

    for v in ranked_videos:
        if cur_dur >= target_duration: break
        fname = f"bg_{prefix_idx}_{len(files)}.mp4"
        print(f"   📦 DL {v['dur']:.1f}s '{v['source']}' Video | desc: '{v['desc']}'", flush=True)
        try:
            resp = requests.get(v['url'], stream=True, headers={"User-Agent": "Mozilla/5.0"})
            with open(fname, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    if chunk: f.write(chunk)
            if os.path.getsize(fname) > 10000:
                files.append(fname)
                cur_dur += v['dur']
        except: pass

    if not files: raise Exception("Failed to download any background videos.")
    return files
