import os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import moviepy.video.fx.all as vfx

from vid_engine import context
from vid_engine.utils import parse_vtt, find_word_timing, CleanLogger
from vid_engine.assets import get_giphy_gif, make_popup, scrape_wikipedia_image, get_background_videos
from vid_engine.templates.base import BaseTemplate

class DefaultTemplate(BaseTemplate):
    
    def build_layer(self, vid_clip, subs, res, popups):
        W, H = res
        fs = int(W * context.FONT_SCALE)
        fp = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if not os.path.exists(fp): fp = "arial.ttf"

        groups, cur = [],[]
        for i, s in enumerate(subs):
            cur.append(s)
            words_left = len(subs) - i - 1

            hit_limit = len(cur) >= 12
            hit_comma = s['text'].endswith(',')
            hit_end = s['text'].endswith(('.', '?', '!'))

            if hit_end:
                groups.append(cur)
                cur =[]
            elif (hit_limit or hit_comma):
                if words_left <= 5 and len(cur) < 18: pass
                else:
                    groups.append(cur)
                    cur =[]
        if cur: groups.append(cur)

        def process_frame(get_frame, t):
            frame = get_frame(t)

            ag = None
            for g in groups:
                if g[0]['start'] - 0.05 <= t <= g[-1]['end'] + 0.15:
                    ag = g
                    break

            active_popups = [p for p in popups if p['start'] - 0.2 <= t < p['end'] + 0.2]
            active_cards = active_popups[:2]

            if not ag and not active_cards: return frame

            img = Image.fromarray(frame).convert("RGBA")
            lyr = Image.new('RGBA', img.size, (255, 255, 255, 0))
            d = ImageDraw.Draw(lyr)

            lines, current_line = [],[]
            try: font = ImageFont.truetype(fp, fs)
            except: font = ImageFont.load_default()

            if ag:
                for w in ag:
                    current_line.append(w)
                    tw = sum(d.textlength(cw['text']+" ", font=font) for cw in current_line)
                    if tw > W * 0.85 and len(current_line) > 1:
                        lines.append(current_line[:-1])
                        current_line = [w]
                if current_line: lines.append(current_line)

            text_total_h = len(lines) * (fs * 1.3) if lines else 0
            card_scale = 0.85 if len(active_cards) > 1 else 1.0

            target_frames =[]
            for p in active_cards:
                fade_in = min(1.0, max(0.0, (t - (p['start'] - 0.2)) / 0.2))
                fade_out = min(1.0, max(0.0, ((p['end'] + 0.2) - t) / 0.2))
                raw_int = min(fade_in, fade_out)

                p['intensity'] = (raw_int ** 2) * (3 - 2 * raw_int)

                af = p['img']['frames'][0]
                fw, fh = max(1, int(af.width * card_scale)), max(1, int(af.height * card_scale))
                target_frames.append((fw, fh))
                p['target_w'] = fw
                p['target_h'] = fh

            layout_type = 'grid'
            if len(target_frames) == 2:
                w1, h1 = target_frames[0]
                w2, h2 = target_frames[1]
                if w1 > h1 and w2 > h2:
                    layout_type = 'stack'

            current_items = []
            for p in active_cards:
                if p['intensity'] > 0:
                    current_items.append({
                        'cw': p['target_w'] * p['intensity'],
                        'ch': p['target_h'] * p['intensity'],
                        'intensity': p['intensity']
                    })

            space_needed = 0
            if current_items:
                avg_int = sum(item['intensity'] for item in current_items) / len(current_items)
                gap_px = 20 * avg_int

                if layout_type == 'stack':
                    space_needed = sum(item['ch'] for item in current_items) + gap_px * (len(current_items) - 1)
                else:
                    space_needed = max(item['ch'] for item in current_items)

            has_cards = space_needed > 0
            max_int_all = max((item['intensity'] for item in current_items), default=0)

            dynamic_gap = 40 * max_int_all if (text_total_h > 0 and has_cards) else 0
            total_block_h = space_needed + dynamic_gap + text_total_h
            start_y = (H - total_block_h) / 2

            if active_cards:
                drawn_active = [p for p in active_cards if p['intensity'] > 0.01]
                draw_gap_px = 20 * (sum(p['intensity'] for p in drawn_active) / len(drawn_active)) if drawn_active else 0

                items_to_draw = []
                for p in sorted(drawn_active, key=lambda x: x['start']):
                    intensity = p['intensity']
                    anim = p['img']
                    elapsed = t - p['start']
                    frame_idx = int(max(0, elapsed) * anim['fps']) % len(anim['frames']) if anim['type'] == 'animated' else 0
                    af = anim['frames'][frame_idx].copy().convert("RGBA")

                    if intensity < 1.0:
                        alpha = af.split()[3]
                        alpha = alpha.point(lambda p_val: int(p_val * intensity))
                        af.putalpha(alpha)

                    current_scale = card_scale * intensity
                    new_w = max(1, int(af.width * current_scale))
                    new_h = max(1, int(af.height * current_scale))
                    df = af.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    items_to_draw.append(df)

                if layout_type == 'stack':
                    cy = start_y
                    for df in items_to_draw:
                        cx = (W - df.width) / 2
                        lyr.paste(df, (int(cx), int(cy)), df)
                        cy += df.height + draw_gap_px
                else:
                    total_w = sum(df.width for df in items_to_draw) + draw_gap_px * (len(items_to_draw) - 1)
                    max_h_current = max((df.height for df in items_to_draw), default=0)
                    cx = (W - total_w) / 2
                    for df in items_to_draw:
                        cy = start_y + (max_h_current - df.height) / 2
                        lyr.paste(df, (int(cx), int(cy)), df)
                        cx += df.width + draw_gap_px

            if ag:
                text_y = start_y + space_needed + dynamic_gap
                for line in lines:
                    lw = sum(d.textlength(cw['text']+" ", font=font) for cw in line)
                    x = (W - lw) / 2
                    for w in line:
                        is_active = (w['start'] <= t <= w['end'])
                        col = (255, 223, 0, 255) if is_active else (255, 255, 255, 255)

                        sk = max(2, int(W * 0.005))
                        for dx, dy in[(-sk,0), (sk,0), (0,-sk), (0,sk), (-sk,-sk), (sk,sk), (-sk,sk), (sk,-sk)]:
                            d.text((int(x + dx), int(text_y + dy)), w['text'], font=font, fill=(0, 0, 0, 255))
                        d.text((int(x), int(text_y)), w['text'], font=font, fill=col)
                        x += d.textlength(w['text'] + " ", font=font)
                    text_y += fs * 1.3

            return np.array(Image.alpha_composite(img, lyr).convert("RGB"))

        return vid_clip.fl(process_frame)

    def make_scene(self, sentence_text, idx, res, kw=None):
        W, H = res
        print(f"\n--- Scene {idx+1} ---", flush=True)
        af, sf = f"a{idx}.mp3", f"s{idx}.vtt"
        print("   🗣️ Voiceover...", flush=True)
        subprocess.run(['edge-tts','--text',sentence_text,f'--rate={context.TTS_RATE}',f'--pitch={context.TTS_PITCH}',f'--volume={context.TTS_VOLUME}','--write-media',af,'--write-subtitles',sf,'--voice',context.VOICE], check=True)

        aud = AudioFileClip(af)
        subs = parse_vtt(sf)
        total_dur = aud.end

        if kw:
            bgs = kw.get("bg_keywords", ["ocean"])
            pps = kw.get("gifs", [])
            wks = kw.get("wiki",[])

            bgs_str = bgs[0] if bgs else "none"
            pps_list =[f"{p.get('keyword', '')}:{p.get('search_query', '')}" for p in pps if p.get('keyword')]
            wks_list =[f"{w.get('keyword', '')}:{w.get('search', '')}" for w in wks if w.get('keyword')]

            print(f"   🧠 {len(bgs)} Keyword ({bgs_str}) | {len(pps_list)} Giphy GIFs ({', '.join(pps_list)}) | {len(wks_list)} Wikipedia Images mapped ({', '.join(wks_list)})", flush=True)
        else:
            bgs = ["nature"]
            pps, wks = [], []

        popups = []

        valid_wks =[w for w in wks if w.get("search") and w.get("keyword")]
        valid_pps = [p for p in pps if p.get("keyword") and p.get("search_query")]

        mapped_pps =[]
        for em_dict in valid_pps:
            pw = em_dict.get("keyword", "")
            sq = em_dict.get("search_query", pw)
            t0, t1 = find_word_timing(pw, subs)
            mapped_pps.append((t0, t1, pw, sq, sentence_text))
        mapped_pps.sort(key=lambda x: x[0])

        for i, (t0, t1, pw, sq, stext) in enumerate(mapped_pps):
            bound_end = total_dur
            if i + 1 < len(mapped_pps):
                bound_end = mapped_pps[i+1][0]

            gif_result = get_giphy_gif(sq, stext)
            if gif_result:
                final_path, gif_title = gif_result
                anim_data = make_popup(final_path, is_wiki=False)
                if anim_data:
                    loop_dur = 1.0
                    if anim_data['type'] == 'animated':
                        loop_dur = len(anim_data['frames']) / anim_data['fps']

                    end_t = min(max(t0 + loop_dur, t1), bound_end)
                    popups.append({"img": anim_data, "start": t0, "end": end_t, "type": "gif", "id": pw})
                    print(f"   🖼️ Overlay Giphy '{pw}:{gif_title}' active for {(end_t - t0):.2f}s", flush=True)

        mapped_wks =[]
        for wk_dict in valid_wks:
            pw = wk_dict.get("keyword", "")
            query = wk_dict.get("search", "")
            t0, t1 = find_word_timing(pw, subs)
            mapped_wks.append((t0, pw, query))

        for (t0, pw, query) in mapped_wks:
            wiki_img_path = scrape_wikipedia_image(query)
            if wiki_img_path:
                anim_data = make_popup(wiki_img_path, is_wiki=True, card_label=query)
                if anim_data:
                    popups.append({"img": anim_data, "start": t0, "end": total_dur, "type": "wiki", "id": query})
                    print(f"   🏛️ Overlay Wiki '{query}' active from {t0:.2f}s to End", flush=True)

        vid_files = get_background_videos(bgs, total_dur, idx, sentence_context=sentence_text)
        if subs: subs[-1]['end'] = total_dur

        print(f"   🎞️ Assembling {total_dur:.2f}s Scene Layer...", flush=True)
        processed =[]
        ta = W/H
        for f in vid_files:
            try:
                c = VideoFileClip(f)
                w,h = c.size; va = w/h
                if va > ta: c = c.crop(x_center=w/2, y_center=h/2, width=int(h*ta), height=h)
                else: c = c.crop(x_center=w/2, y_center=h/2, width=w, height=int(w/ta))
                processed.append(c.resize(res))
            except: pass

        vid = concatenate_videoclips(processed, method="compose")
        vid = vid.fx(vfx.loop, duration=total_dur) if vid.duration < total_dur else vid.subclip(0, total_dur)

        vid = vid.set_audio(aud)
        final_clip = self.build_layer(vid, subs, res, popups)

        scene_file = f"rendered_scene_{idx}.mp4"
        my_log = CleanLogger(int(final_clip.duration * 24))
        try:
            final_clip.write_videofile(scene_file, fps=24, codec=context.VIDEO_CODEC, audio_codec="aac", logger=my_log, threads=8, preset="ultrafast")
        except:
            my_log.close(); my_log = CleanLogger(int(final_clip.duration * 24))
            final_clip.write_videofile(scene_file, fps=24, codec="libx264", audio_codec="aac", logger=my_log, threads=8, preset="ultrafast")
        finally:
            my_log.close()

        try:
            aud.close()
            vid.close()
            final_clip.close()
            for p in processed: p.close()
        except: pass

        return scene_file, total_dur
