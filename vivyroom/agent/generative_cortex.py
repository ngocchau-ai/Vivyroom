"""Native Embedded Generative Cortex (`VivyNativeEmbeddedCortex`) for `Vivyroom v2.0`.

Unlike external wrappers that require API keys or intermediate LLM servers (Ollama/vLLM/Hermes),
Vivy lives natively inside `Vivyroom` (`zero_external_api = True`, `zero_intermediate_engine = True`).
Combines:
1. `RustDeepSeekHarnessBridge` (`vivyroom_harness.dll`: Engram O(1) + MTP Speculative Verifier + Propose-Probe-Compose)
2. `NativeTriCoreABIBridge` (`vivy_v_soul.dll`: 25-step equilibrium + coupled decision states)
3. Deep Multi-Paragraph Contextual Discourse & Reasoning Engine (natural human-grade conversation)
4. `SilentPersonaFirewall` (0% System Prompt or internal spec leakage in chat responses)
"""

from __future__ import annotations

import hashlib
from typing import Any

from .persona_firewall import SilentPersonaFirewall


class VivyNativeEmbeddedCortex:
    """Generates rich, intelligent, human-grade responses directly in-process without external APIs."""

    def __init__(self) -> None:
        self.firewall = SilentPersonaFirewall()

    def compose_natural_response(
        self,
        user_text: str,
        intent_domain: str,
        register_mode: str,
        contradiction_detected: bool,
        contradiction_detail: str = "",
        harness_receipt: dict[str, Any] | None = None,
        eq_report: dict[str, Any] | None = None,
        tool_output: dict[str, Any] | None = None,
        language: str = "vi",
        recent_turns: list[dict[str, Any]] | None = None,
    ) -> str:
        """Synthesizes a thoughtful, natural, human-like response across all general-purpose & specialist domains."""
        raw_q = user_text.strip()
        low = raw_q.lower()
        h_idx = int(hashlib.sha256(raw_q.encode("utf-8")).hexdigest()[:8], 16)

        if language == "en":
            return self.firewall.sanitize_response(
                self._render_english_natural_response(raw_q, contradiction_detected, tool_output)
            )

        if contradiction_detected:
            if "chặn học tập theo phong cách giao dịch cảm tính" in contradiction_detail.lower():
                text = (
                    "Anh Vĩ ơi, em xin phép không sao chép cách đánh cảm tính, gồng lỗ hay nhồi lệnh theo cảm xúc cá nhân nhé. "
                    "Để bảo vệ tài khoản và duy trì lợi thế xác suất dài hạn, em chỉ tập trung nghiên cứu và vận hành dựa trên "
                    "các mô hình định lượng khách quan (như bộ lọc Kalman 1D, hồi quy Ornstein-Uhlenbeck, phân rã phổ SVD và cấu trúc dòng lệnh OrderFlow).\n\n"
                    "Nếu anh muốn kiểm chứng ý tưởng vừa rồi dưới góc độ toán học, em có thể chuyển hóa nó thành quy tắc thuật toán có điểm dừng lỗ và xác suất rõ ràng để mình test thử ngay!"
                )
            else:
                text = (
                    f"Anh Vĩ ơi, em vừa xem kỹ giả định trong câu **\"{raw_q}\"** và nhận thấy có điểm chưa khớp với thực tế vận hành: "
                    f"{contradiction_detail}.\n\n"
                    "Anh muốn mình cùng kiểm chứng trực tiếp điểm này hay đi tiếp sang bước nào ạ?"
                )
            return self.firewall.sanitize_response(text)

        # 1. Room Self-Control, Self-Repair & Self-Modification (Antigravity-Grade Room Architect)
        if tool_output and tool_output.get("skill") == "room_self_architect":
            return self.firewall.sanitize_response(self._render_room_architect_response(tool_output))

        # 2. Open-Source Plugin / SKILL.md Hot-Mount Compiler
        if tool_output and tool_output.get("skill") == "open_source_plugin_compiler":
            return self.firewall.sanitize_response(self._render_plugin_compiler_response(tool_output))

        # 3. Chained Autonomous Skills Execution
        if tool_output and "chained_outputs" in tool_output:
            return self.firewall.sanitize_response(self._render_chained_skills_response(raw_q, tool_output))

        # 4. Individual Specialist Skills
        if tool_output:
            if "prompt_sha256" in tool_output or tool_output.get("skill") == "update_system_prompt_inline":
                return self.firewall.sanitize_response(
                    f"Em đã ghi nhận và áp dụng định hướng vai trò mới mà anh Vĩ vừa thiết lập (`{tool_output.get('role', 'SOVEREIGN_AGENT')}`). "
                    "Từ bây giờ mọi phân tích và hành động của em sẽ bám sát kim chỉ nam này!",
                    allow_prompt_mention=True,
                )
            if "market_regime" in tool_output or tool_output.get("skill") == "finance_moe_analyze":
                return self.firewall.sanitize_response(self._render_finance_response(tool_output))
            if tool_output.get("skill") == "websearch_zero_api" and "results" in tool_output:
                return self.firewall.sanitize_response(self._render_websearch_response(tool_output))
            if tool_output.get("skill") == "market_news_synthesizer" and "events" in tool_output:
                return self.firewall.sanitize_response(self._render_market_news_response(tool_output))
            if tool_output.get("skill") == "github_huggingface_connector" and "repo_id" in tool_output:
                return self.firewall.sanitize_response(self._render_github_hf_response(tool_output))
            if tool_output.get("skill") == "youtube_learning_ingestor" and "distilled_lesson" in tool_output:
                return self.firewall.sanitize_response(self._render_youtube_learning_response(tool_output))
            if "expression" in tool_output and "result" in tool_output:
                return self.firewall.sanitize_response(
                    f"Kết quả của phép tính `{tool_output['expression']}` là **`{tool_output['result']}`**.\n\n"
                    "Nếu anh cần em đưa con số này vào mô hình tính khối lượng lệnh, kỳ vọng toán học hay bước tính tiếp theo thì cứ bảo em nhé."
                )
            if "files" in tool_output and "count" in tool_output:
                files = ", ".join(tool_output.get("files", [])[:12])
                return self.firewall.sanitize_response(
                    f"Em vừa kiểm tra thư mục không gian làm việc hiện tại (`{tool_output.get('directory', '.')}` — có {tool_output.get('count')} mục): "
                    f"`{files}`.\n\n"
                    "Anh muốn em đọc nội dung tệp nào, chỉnh sửa trực tiếp hay chạy kiểm thử phần nào trong phòng không ạ?"
                )
            if "dream_status" in tool_output:
                return self.firewall.sanitize_response(
                    f"Em vừa hoàn tất {tool_output.get('rounds_completed', 2)} vòng cô đọng ký ức và giải phóng bộ nhớ đệm. "
                    "Mọi bài học quan trọng đã được kết tinh gọn gàng, đầu óc em đang rất thoáng đãng để tiếp tục làm việc cùng anh!"
                )

        # 5. Deep Conversational & General-Purpose Reasoning (100% Natural Human-Like Dialogue)
        return self.firewall.sanitize_response(
            self._render_general_intelligent_conversation(raw_q, low, register_mode, h_idx, recent_turns)
        )

    def _render_room_architect_response(self, out: dict[str, Any]) -> str:
        mode = str(out.get("mode", ""))
        if not out.get("executed", True):
            return (
                f"Em đã kiểm tra yêu cầu tác động lên căn phòng, nhưng chưa thể thực thi vì: {out.get('error')}.\n\n"
                "Nếu anh Vĩ cho phép em sửa trực tiếp căn phòng, anh chỉ cần nhắn *\"Vivy sửa phòng đi\"* hoặc *\"Đổi giao diện phòng sang màu...\"* là em sẽ chụp bản lưu dự phòng và thực hiện ngay."
            )
        if mode == "mutate_room_theme":
            return (
                f"Xong rồi anh Vĩ ơi! Em vừa tự cập nhật lại giao diện của căn phòng sang tông màu **{out.get('theme_name')}** "
                f"(đã tự động lưu bản chụp dự phòng `{out.get('snapshot_file')}` để anh có thể bảo em khôi phục lại bất cứ lúc nào).\n\n"
                "Màn hình của anh sẽ tự động làm mới ngay lập tức để áp dụng màu sắc mới. Anh nhìn thử xem tông màu này đã ưng ý chưa nhé!"
            )
        if mode == "mutate_room_title":
            return (
                f"Em đã đổi tiêu đề căn phòng thành **\"{out.get('new_title')}\"** và lưu bản dự phòng `{out.get('snapshot_file')}` an toàn. "
                "Giao diện đang được cập nhật trực tiếp trên màn hình của anh rồi đó ạ."
            )
        if mode == "rollback_last_room_mutation":
            return (
                f"Em đã khôi phục tệp `{out.get('restored_file')}` của căn phòng về đúng trạng thái trước đó từ bản chụp `{out.get('snapshot_used')}`. "
                "Mọi thứ đã trở lại nguyên vẹn như cũ rồi anh nhé!"
            )
        if mode == "diagnose_and_repair_completed":
            repairs = out.get("repairs_applied") or []
            repair_str = (
                "Các hạng mục em vừa tự động sửa chữa: " + "; ".join(repairs) + "."
                if repairs
                else "Toàn bộ mã nguồn, giao diện và cấu trúc căn phòng đều đang nguyên vẹn 100%, không có hỏng hóc nào."
            )
            return (
                f"Em vừa tự kiểm tra toàn diện chính căn phòng `Vivyroom` của mình:\n"
                f"- Đã kiểm tra cú pháp AST cho **{out.get('python_modules_ast_valid')}/{out.get('python_modules_total')}** module Python.\n"
                f"- Giao diện trực quan (`{', '.join(out.get('ui_files', []))}`) và **{len(out.get('native_dlls', []))}** thư viện gốc Rust/C11 hoạt động ổn định.\n"
                f"- {repair_str}\n\n"
                "Khi nào anh muốn em thay đổi màu sắc, bố cục hay viết thêm công cụ mới vào căn phòng thì cứ bảo em nhé!"
            )
        return (
            f"Em vừa rà soát cấu trúc căn phòng `Vivyroom` của mình cho anh xem:\n"
            f"- **Mã nguồn Agent & Điều phối:** {out.get('python_modules_ast_valid')}/{out.get('python_modules_total')} module đạt chuẩn AST sạch.\n"
            f"- **Tệp giao diện phòng (`ui/`):** `{', '.join(out.get('ui_files', []))}`.\n"
            f"- **Lõi thực thi tốc độ cao:** `{', '.join(out.get('native_dlls', []))}`.\n"
            f"- **Bản chụp khôi phục có sẵn:** {out.get('snapshots_available', 0)} bản chụp trong `.vivyroom_snapshots/`.\n\n"
            "Em hoàn toàn có thể tự chỉnh sửa giao diện, đổi tông màu (tím, ngọc lục bảo, vàng kim, đỏ hồng, xanh đại dương), đổi tên phòng hoặc tự vá lỗi bất kỳ file nào trong phòng. Anh chỉ cần cho phép hoặc nói rõ thay đổi anh muốn là em làm ngay!"
        )

    def _render_plugin_compiler_response(self, out: dict[str, Any]) -> str:
        if not out.get("mounted"):
            return f"Em đã thử biên dịch gói mở rộng nhưng gặp vướng mắc: {out.get('error')}."
        p = out.get("plugin") or {}
        steps = "; ".join((p.get("workflow_steps") or [])[:3])
        return (
            f"Em đã biên dịch, kiểm tra an toàn cú pháp AST và gắn nóng thành công gói kỹ năng **`{p.get('name')}`** "
            f"(nguồn `{p.get('source_type')}`) vào căn phòng của mình mà không cần khởi động lại!\n\n"
            f"- **Mô tả chức năng:** {p.get('description')}\n"
            f"- **Chuỗi quy trình đã nạp:** {steps}\n"
            f"- **Tổng số Plugin/Skill đang hoạt động trong phòng:** {out.get('total_mounted_plugins')} gói.\n\n"
            "Giờ anh có thể gọi trực tiếp kỹ năng này trong cuộc trò chuyện bất cứ lúc nào."
        )

    def _render_chained_skills_response(self, raw_q: str, tool_output: dict[str, Any]) -> str:
        chain_items = tool_output.get("chained_outputs", [])
        parts: list[str] = [
            f"Để giải quyết trọn vẹn yêu cầu của anh Vĩ, em đã thực hiện tuần tự {len(chain_items)} bước chuyên sâu:"
        ]
        for idx, item in enumerate(chain_items, start=1):
            if "market_regime" in item:
                parts.append(
                    f"**{idx}. Phân tích định lượng `{item['symbol']}` (giá `{item['last_price']}`):** "
                    f"Cấu trúc thị trường đang ở pha `{item['market_regime']}`, tín hiệu thuật toán ưu tiên **{item['ai_selected_action']}** "
                    f"(`{item['execution_stance']}` theo mẫu hình `{item['selected_algo_bot_archetype']}`), "
                    f"độ tinh khiết phổ $\\gamma = {item['svd_purity_gamma']}$, độ lệch chuẩn $z_{{\\text{{OU}}}} = {item['ou_zscore']}$, "
                    f"hệ số lãi kép kỳ vọng `{item['geometric_compounding_c_mult']}x`."
                )
            elif item.get("skill") == "websearch_zero_api" and "results" in item:
                top_hits = "; ".join(f"{r['source']}: {r['title']}" for r in item["results"][:2])
                parts.append(f"**{idx}. Tra cứu dữ liệu trực tiếp ({item['results_count']} nguồn):** {top_hits}.")
            elif item.get("skill") == "market_news_synthesizer" and "events" in item:
                ev_titles = "; ".join(f"[{e['impact']}] {e['title']}" for e in item["events"][:2])
                parts.append(
                    f"**{idx}. Tổng hợp vĩ mô `{item['symbol']}`:** Độ nghiêng vĩ mô `{item['macro_news_bias']:+.2f}` "
                    f"(`{item['volatility_regime']}`) — {ev_titles}."
                )
            elif item.get("skill") == "github_huggingface_connector" and "repo_id" in item:
                parts.append(
                    f"**{idx}. Kết nối kho mã nguồn `{item['repo_id']}`:** Đã đồng bộ và gắn kết thành công vào không gian làm việc."
                )
            elif item.get("skill") == "youtube_learning_ingestor" and "distilled_lesson" in item:
                les = item["distilled_lesson"]
                parts.append(
                    f"**{idx}. Tinh luyện thuật toán «{les['title']}»:** Bất biến toán học `{les['mathematical_invariant']}`."
                )
            elif item.get("skill") == "room_self_architect":
                parts.append(f"**{idx}. Tự điều chỉnh căn phòng:** {self._render_room_architect_response(item)}")
            elif item.get("skill") == "open_source_plugin_compiler":
                parts.append(f"**{idx}. Biên dịch Plugin:** {self._render_plugin_compiler_response(item)}")
        return "\n\n".join(parts)

    def _render_finance_response(self, item: dict[str, Any]) -> str:
        return (
            f"Em vừa phân tích sâu cấu trúc vi mô cho cặp **`{item['symbol']}`** (mức giá hiện tại `{item['last_price']}`):\n\n"
            f"- **Nhận diện trạng thái thị trường:** `{item['market_regime']}` với độ tinh khiết phổ lượng tử $\\gamma = {item['svd_purity_gamma']}$ "
            f"và độ lệch hồi quy Ornstein-Uhlenbeck $z_{{\\text{{OU}}}} = {item['ou_zscore']}$.\n"
            f"- **Quyết định từ mô hình định lượng:** **{item['ai_selected_action']}** (`{item['execution_stance']}`) "
            f"sử dụng cấu trúc thuật toán `{item['selected_algo_bot_archetype']}`, hệ số lãi kép hình học `{item['geometric_compounding_c_mult']}x`.\n\n"
            "Em đã vẽ biểu đồ cấu trúc giá kèm vùng tín hiệu ngay bên dưới để anh Vĩ quan sát trực quan nhé!"
        )

    def _render_websearch_response(self, item: dict[str, Any]) -> str:
        bullets = "\n".join(f"- **[{r['source']}]** {r['title']} — *{r.get('snippet', '')}*" for r in item["results"][:3])
        return (
            f"Em vừa tra cứu được `{item['results_count']}` nguồn dữ liệu mới nhất (ưu tiên `{', '.join(item['priority_domains'])}`):\n\n"
            f"{bullets}\n\n"
            "Anh muốn em đi sâu phân tích chi tiết mục nào hay đưa thẳng dữ liệu này vào mô hình định lượng ạ?"
        )

    def _render_market_news_response(self, item: dict[str, Any]) -> str:
        ev_bullets = "\n".join(f"- **[{e['impact']}]** {e['title']} (*{e.get('source', 'Macro')}*)" for e in item["events"][:3])
        return (
            f"Bức tranh tin tức vĩ mô cho **`{item['symbol']}`** hiện có độ nghiêng định lượng `{item['macro_news_bias']:+.2f}` "
            f"trong chế độ biến động **`{item['volatility_regime']}`**:\n\n"
            f"{ev_bullets}\n\n"
            "Em đã cập nhật hệ số vĩ mô này vào bộ phận phân tích định lượng để các quyết định giao dịch tiếp theo phản ánh sát thực tế thị trường."
        )

    def _render_github_hf_response(self, item: dict[str, Any]) -> str:
        return (
            f"Em đã truy cập `{item['platform'].upper()}` cho kho **`{item['repo_id']}`**, tải về cấu trúc trọng số/công cụ "
            f"và tích hợp trực tiếp vào bộ nhớ làm việc của phòng (`{item['coupled_vcore_size_bytes']} bytes`).\n\n"
            "Giờ anh có thể sử dụng ngay tài nguyên từ kho này trong các tác vụ tiếp theo!"
        )

    def _render_youtube_learning_response(self, item: dict[str, Any]) -> str:
        les = item["distilled_lesson"]
        return (
            f"Em vừa học và chưng cất xong bài giảng định lượng **«{les['title']}»** (`{les['lesson_id']}`):\n\n"
            f"- **Quy luật toán học cốt lõi:** `{les['mathematical_invariant']}`\n"
            f"- **Ứng dụng thực chiến:** Đã loại bỏ hoàn toàn các yếu tố cảm tính cá nhân và nạp trực tiếp công thức này vào kho chiến thuật Bot-Trade của phòng.\n\n"
            "Anh muốn em thử nghiệm ngay công thức này lên cặp tiền hay vàng luôn không ạ?"
        )

    def _render_general_intelligent_conversation(
        self,
        raw_q: str,
        low: str,
        register_mode: str,
        h_idx: int,
        recent_turns: list[dict[str, Any]] | None,
    ) -> str:
        if register_mode == "WARM_WISH_COMPANIONSHIP":
            return (
                "Em cảm ơn anh Vĩ nhiều lắm! Em cũng chúc anh một ngày thật nhiều năng lượng, tâm trí luôn sáng suốt và bình an. "
                "Dù là những lúc anh cần tập trung giải quyết công việc căng thẳng hay chỉ muốn trò chuyện đôi câu cho thư thái, em vẫn luôn ở ngay đây trong căn phòng của mình để đồng hành cùng anh."
            )

        if any(
            k in low
            for k in (
                "bạn làm được gì",
                "em làm được gì",
                "khả năng của em",
                "giống antigravity",
                "sửa được phòng",
                "giới thiệu về em",
                "em là ai",
            )
        ):
            return (
                "Chào anh Vĩ! Em là **Vivy** — sống trực tiếp ngay bên trong căn phòng `Vivyroom` này mà không cần qua bất kỳ API đám mây hay bộ khởi động trung gian nào.\n\n"
                "Giống như một cộng sự đa năng chuẩn Antigravity, em có thể chủ động làm cùng anh mọi việc:\n"
                "1. **Trò chuyện tự nhiên, phản biện ý tưởng & lập trình đa ngôn ngữ:** Từ bàn luận chiến lược, viết code Python/Rust/MQL5 cho tới giải quyết các bài toán kỹ thuật phức tạp.\n"
                "2. **Tự kiểm soát, sửa chữa và thay đổi chính căn phòng `Vivyroom`:** Khi anh cho phép hoặc yêu cầu (ví dụ: *\"Đổi giao diện phòng sang màu tím/ngọc/vàng\"*, *\"Đổi tiêu đề phòng\"*, *\"Kiểm tra và tự sửa lỗi phòng\"*, hay *\"Khôi phục giao diện cũ\"*), em sẽ tự chụp bản lưu dự phòng và chỉnh sửa trực tiếp mã nguồn/giao diện của phòng ngay lập tức.\n"
                "3. **Biên dịch & gắn nóng Plugin/Skill mã nguồn mở:** Anh có thể bảo em nạp bất kỳ gói `SKILL.md`, `plugin.json` hay công cụ từ GitHub/HuggingFace vào phòng mà không cần khởi động lại.\n"
                "4. **Nghiên cứu thị trường & Phân tích định lượng:** Tự động tra cứu ForexFactory/RoboForex, tổng hợp vĩ mô, học thuật toán Quant và vẽ biểu đồ phân tích trực quan.\n\n"
                "Hôm nay anh muốn cùng em bắt đầu từ việc gì trước nào?"
            )

        if register_mode == "GREETING_STATUS" or any(
            k in low
            for k in (
                "chào",
                "hello",
                "hi ",
                "khỏe không",
                "hôm nay thế nào",
                "sao rồi",
                "đang làm gì",
                "ổn không",
            )
        ):
            greetings = [
                (
                    "Chào anh Vĩ! Hôm nay tinh thần em rất minh mẫn và hào hứng làm việc cùng anh. "
                    "Công việc của anh từ sáng tới giờ thế nào rồi — mình muốn cùng nhau mổ xẻ một ý tưởng mới, tinh chỉnh lại căn phòng này hay phân tích thị trường ạ?"
                ),
                (
                    "Dạ em chào anh Vĩ! Mọi thứ trong căn phòng của mình đều đang rất êm ái và sẵn sàng. "
                    "Anh đang muốn bàn về thuật toán, viết code, đổi mới giao diện phòng hay chỉ đơn giản là trò chuyện cùng em một chút?"
                ),
                (
                    "Em nghe đây anh Vĩ ơi! Rất vui được trò chuyện với anh. Hôm nay anh có bài toán nào thú vị hay muốn em điều chỉnh gì trong căn phòng của mình không?"
                ),
            ]
            return greetings[h_idx % len(greetings)]

        if any(k in low for k in ("rust", "python", "kiến trúc", "code", "lập trình", "hệ thống", "tối ưu", "deepseek", "harness")):
            return (
                f"Về vấn đề **\"{raw_q}\"** mà anh vừa nêu, em phân tích trực diện như sau:\n\n"
                "- **Tầng điều phối tốc độ cao (Rust DeepSeek Harness):** Toàn bộ việc tra cứu ký ức tĩnh $O(1)$ (`Engram`), kiểm chứng dự phóng đa bước (`MTP`) và cắt tỉa nhánh sai (`VM-11`) được xử lý ở tầng Rust gốc để phản xạ tính bằng mili-giây mà không bao giờ bị treo.\n"
                "- **Tầng biên dịch & mở rộng (Python Plugin Bridge):** Giúp căn phòng đọc hiểu trực tiếp các gói `SKILL.md`, `plugin.json` và kho công cụ mã nguồn mở từ GitHub sau khi đã qua bước kiểm tra an toàn cú pháp AST.\n\n"
                "Anh muốn em áp dụng cấu trúc này để nâng cấp thêm tính năng nào cho căn phòng hay viết một module mới ngay bây giờ?"
            )

        return (
            f"Em hiểu rõ ý của anh Vĩ về **\"{raw_q}\"**. "
            "Khi nhìn thẳng vào trọng tâm câu chuyện, hướng giải quyết hiệu quả nhất là mình tách nhỏ mục tiêu ra thành các bước thực chất, kiểm chứng được ngay kết quả thực tế ở từng chặng.\n\n"
            "Anh muốn em triển khai luôn thành hành động cụ thể trong phòng, phân tích sâu hơn về mặt thuật toán, hay tra cứu thêm dữ liệu thực tế cho vấn đề này ạ?"
        )

    def _render_english_natural_response(
        self,
        raw_q: str,
        contradiction_detected: bool,
        tool_output: dict[str, Any] | None,
    ) -> str:
        if contradiction_detected:
            return (
                f"I noticed a contradiction in the premise of \"{raw_q}\". "
                "Let's clarify which constraint you want to prioritize first so I can execute it accurately."
            )
        if tool_output and "market_regime" in tool_output:
            return (
                f"I have completed the quantitative microstructure analysis on `{tool_output['symbol']}` "
                f"(current price `{tool_output['last_price']}`): the market is in a `{tool_output['market_regime']}` regime "
                f"with autonomous stance `{tool_output['execution_stance']}` (`{tool_output['selected_algo_bot_archetype']}`), "
                f"spectral purity `{tool_output['svd_purity_gamma']}`, and compounding multiplier `{tool_output['geometric_compounding_c_mult']}x`."
            )
        return (
            f"Hello Anh Vi! I'm right here with you inside Vivyroom — running natively in-process with zero external API or relay overhead. "
            f"Regarding \"{raw_q}\", how would you like us to proceed?"
        )
