"""Run standalone reply-merge regressions with fake host and provider objects.

The real thinking, scheduling, history and lifecycle hooks run without network
access. Execute with ``python -B tests/test_merge_logic.py``.
"""

import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent.parent / "main.py"


def install_stubs() -> None:
    """Install minimal astrbot stand-ins so main.py imports standalone."""
    pkg = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    api.AstrBotConfig = dict

    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = type("AstrMessageEvent", (), {})

    class _Filter:
        EventMessageType = type(
            "EventMessageType", (), {"ALL": "all", "GROUP_MESSAGE": "group"}
        )

        def event_message_type(self, *_a, **_k):
            return lambda f: f

        def on_waiting_llm_request(self, *_a, **_k):
            return lambda f: f

        def on_llm_request(self, *_a, **_k):
            return lambda f: f

        def on_llm_response(self, *_a, **_k):
            return lambda f: f

        def on_astrbot_loaded(self, *_a, **_k):
            return lambda f: f

        def command(self, *_a, **_k):
            return lambda f: f

    event.filter = _Filter()

    comps = types.ModuleType("astrbot.api.message_components")
    for name in ("At", "File", "Forward", "Image", "Record", "Video"):
        setattr(comps, name, type(name, (), {}))

    platform = types.ModuleType("astrbot.api.platform")
    platform.MessageType = type("MessageType", (), {"OTHER_MESSAGE": "other_message"})

    provider = types.ModuleType("astrbot.api.provider")
    provider.ProviderRequest = type("ProviderRequest", (), {})
    provider.LLMResponse = type("LLMResponse", (), {})

    agent_message = types.ModuleType("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text):
            self.text = text
            self.temporary = False

        def mark_as_temp(self):
            self.temporary = True
            return self

    agent_message.TextPart = TextPart

    star = types.ModuleType("astrbot.api.star")
    star.Context = type("Context", (), {})

    class _Star:
        def __init__(self, context):
            self.context = context

    star.Star = _Star

    pkg.api = api
    sys.modules["astrbot"] = pkg
    for mod in (api, event, comps, platform, provider, agent_message, star):
        sys.modules[mod.__name__] = mod


def load_plugin():
    spec = importlib.util.spec_from_file_location("reply_merge_main", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reply_merge_main"] = module
    spec.loader.exec_module(module)
    return module


class FakeEvent:
    """Provide the host event interface used by the real plugin hooks."""

    def __init__(
        self,
        umo="aiocqhttp:GroupMessage:1",
        sender="u1",
        private=False,
        text="hello",
        images=(),
        order=0,
        wake=True,
    ):
        self.unified_msg_origin = umo
        self._sender = sender
        self._private = private
        self._text = text
        self._images = list(images)
        self._stopped = False
        self.call_llm = False
        self.is_at_or_wake_command = wake
        self.message_obj = types.SimpleNamespace(
            session_id=umo.split(":")[-1], timestamp=order
        )
        self._extra = {}

    def get_sender_id(self):
        return self._sender

    def is_private_chat(self):
        return self._private

    def get_sender_name(self):
        return "Test sender"

    def get_message_type(self):
        return "group_message"

    def get_platform_name(self):
        return self.unified_msg_origin.split(":", 1)[0]

    def should_call_llm(self, call_llm):
        self.call_llm = call_llm

    def get_result(self):
        return None

    def stop_event(self):
        self._stopped = True

    def is_stopped(self):
        return self._stopped

    def get_message_str(self):
        return self._text

    def get_message_outline(self):
        return self._text

    def get_messages(self):
        image_type = sys.modules["astrbot.api.message_components"].Image
        images = []
        for source in self._images:
            image = image_type()
            image.url = source
            image.file = ""
            images.append(image)
        return images

    def get_extra(self, key, default=None):
        return self._extra.get(key, default)

    def set_extra(self, key, value):
        self._extra[key] = value


def test_window_grouping(m):
    w = m.MergeWindow(
        5.0,
        60.0,
        first_message="leader text",
        leader_order=(1, 0, 0),
        leader_images=["img_leader"],
    )
    assert w.unseen == 0 and w.close_reason == "" and w.rescued is False
    w.donate((0, 0, 9), "sent before leader", [], [])
    w.donate((2, 0, 1), "sent after", ["img_after"], ["语音消息"])
    w.donate((3, 0, 1), "", ["sticker"], [])
    before, after = w.ordered_messages()
    assert [message.lines for message in before] == [["- sent before leader"]], before
    assert [message.lines for message in after] == [
        ["- sent after", "- [语音消息]"],
        ["- [图片/表情包]"],
    ], after
    assert all(isinstance(message, m.BufferedMessage) for message in w.buffer)
    # One buffer entry per message, and unseen counts donated messages.
    assert len(w.buffer) == 3 and w.unseen == 3
    w.donate((4, 0, 1), "newest", ["img_leader"], [])
    assert w.unseen == 4
    assert w.all_images() == ["img_leader", "img_after", "sticker"]
    # Truncation keeps whole messages and the newest ones.
    _, after = w.ordered_messages()
    shown = after[-2:]
    omitted = len(after) - len(shown)
    assert omitted == 1
    assert [message.lines for message in shown] == [
        ["- [图片/表情包]"],
        ["- newest"],
    ], shown
    print("grouping/ordering OK")


def test_verdict_parsing(m):
    v = m.parse_verdict("想到什么算什么\n<<wait 12>>", True, 7.0)
    assert v is not None and v[0] == "想到什么算什么" and v[4:] == ("wait", 12.0), v
    v = m.parse_verdict("说完了 <<REPLY>>", True, 7.0)
    assert v[4] == "reply" and v[5] == m.THINK_REPLY_PAUSE, v
    v = m.parse_verdict("纯独白没有决定行", True, 7.0)
    assert v[4] == "wait" and v[5] == 7.0, v
    v = m.parse_verdict('{"thought":"心里话","action":"wait","wait":18}', True, 7.0)
    assert v[0] == "心里话" and v[5] == 18.0, v
    assert m.parse_verdict("", True, 7.0) is None
    v = m.parse_verdict(
        '{"points":"一句话","summary":"说完了","questions":[],'
        '"action":"reply","wait":0}',
        False,
        7.0,
    )
    assert v is not None and v[1] == ["一句话"] and v[4] == "reply", v
    print("verdict parsing OK")


def test_scope_gate(m):
    plugin = m.ReplyMergePlugin(None, {})
    group_ev = FakeEvent(umo="aiocqhttp:GroupMessage:1")
    private_ev = FakeEvent(umo="aiocqhttp:FriendMessage:2", private=True)
    assert plugin._debounce_enabled(group_ev)
    assert plugin._debounce_enabled(private_ev)

    both_off = m.ReplyMergePlugin(None, {"usage_scope": "group"})
    assert both_off._debounce_enabled(group_ev)
    assert not both_off._debounce_enabled(private_ev)
    priv_only = m.ReplyMergePlugin(None, {"usage_scope": "private"})
    assert not priv_only._debounce_enabled(group_ev)
    assert priv_only._debounce_enabled(private_ev)

    by_umo = m.ReplyMergePlugin(
        None, {"disabled_sessions": ["aiocqhttp:GroupMessage:1"]}
    )
    assert not by_umo._debounce_enabled(group_ev)
    assert by_umo._debounce_enabled(FakeEvent(umo="aiocqhttp:GroupMessage:9"))
    by_session = m.ReplyMergePlugin(None, {"disabled_sessions": ["1"]})
    assert not by_session._debounce_enabled(group_ev)

    by_user = m.ReplyMergePlugin(None, {"disabled_users": ["u1"]})
    assert not by_user._debounce_enabled(group_ev)
    assert by_user._debounce_enabled(FakeEvent(sender="u2"))
    print("scope gate OK")


def test_stale_reply_discard(m):
    plugin = m.ReplyMergePlugin(None, {})
    old_window = m.MergeWindow(5.0, 60.0, first_message="old")
    other_window = m.MergeWindow(5.0, 60.0, first_message="other")
    new_window = m.MergeWindow(5.0, 60.0, first_message="new")
    key = "aiocqhttp:GroupMessage:1:u1"
    stale_event = FakeEvent()
    stale_event.set_extra("_reply_merge_from_window", old_window)
    plugin.inflight[key] = (old_window, 0.0)
    # Two stale windows may be pending at once; blanking one keeps the
    # other marked.
    plugin.stale[key] = {old_window, other_window}

    class Resp:
        completion_text = "old reply"

    resp = Resp()
    asyncio.run(plugin.drop_stale_reply(stale_event, resp))
    assert resp.completion_text == "", resp.completion_text
    assert key not in plugin.inflight
    assert plugin.stale[key] == {other_window}

    # A live (non-stale) reply passes through untouched.
    live_event = FakeEvent()
    live_event.set_extra("_reply_merge_from_window", new_window)
    resp = Resp()
    asyncio.run(plugin.drop_stale_reply(live_event, resp))
    assert resp.completion_text == "old reply"
    # Events without our marker are ignored entirely.
    resp = Resp()
    asyncio.run(plugin.drop_stale_reply(FakeEvent(), resp))
    assert resp.completion_text == "old reply"
    print("stale reply discard OK")


def test_stale_content_rides(m):
    """A window opened over an in-flight request carries its content.

    The blanked reply's whole turn is dropped from the host history, so
    the stale burst must ride with the newer window or it is never
    answered at all.
    """
    settle, pause = m.THINK_SETTLE_SECONDS, m.THINK_REPLY_PAUSE
    m.THINK_SETTLE_SECONDS = 0.05
    m.THINK_REPLY_PAUSE = 0.05
    try:
        plugin = m.ReplyMergePlugin(
            None,
            {
                "min_wait": 1,
                "max_wait": 2,
                "max_window": 30,
                "enable_thinking": True,
                "think_context": False,
                "think_style": "",
            },
        )
        event = FakeEvent()
        key = m.merge_key(event)
        inflight_window = m.MergeWindow(
            1.0, 30.0, first_message="stale leader", leader_images=["img_stale"]
        )
        inflight_window.donate((1, 0, 1), "stale follow-up", [], [])
        plugin.inflight[key] = (inflight_window, time.monotonic())

        async def fake_think(
            event, first_message, window, elapsed=0.0, remaining=0.0, style_context=""
        ):
            return m.Verdict([], "", [], "reply", 0.0)

        plugin._think = fake_think
        asyncio.run(plugin.hold_window(event))

        new_window = event.get_extra("_reply_merge_window")
        assert isinstance(new_window, m.MergeWindow)
        assert plugin.stale[key] == {inflight_window}
        # The stale burst rides whole: leader text and follow-up, plus its
        # image, emptied out of the dead window so nothing answers twice.
        assert [message.lines for message in new_window.earlier] == [
            ["- stale leader"],
            ["- stale follow-up"],
        ], new_window.earlier
        assert len(new_window.earlier) == 2
        assert new_window.all_images() == ["img_stale"]
        assert inflight_window.rescued is True
        assert not inflight_window.buffer and inflight_window.first_message == ""
    finally:
        m.THINK_SETTLE_SECONDS = settle
        m.THINK_REPLY_PAUSE = pause
    print("stale content rides OK")


def test_lazy_loop(m):
    """Drive the real leader loop with a stubbed think pass.

    A burst of leader + one follow-up must cost exactly two passes: the
    eager first pass and the one final pass at the deadline — the stray
    follow-up must NOT trigger a pass of its own, and the window must
    still close on the quiet deadline with everything read.
    """
    settle, pause = m.THINK_SETTLE_SECONDS, m.THINK_REPLY_PAUSE
    m.THINK_SETTLE_SECONDS = 0.05
    m.THINK_REPLY_PAUSE = 0.05
    try:
        plugin = m.ReplyMergePlugin(
            None,
            {
                "min_wait": 1,
                "max_wait": 2,
                "max_window": 30,
                "enable_thinking": True,
                "think_context": False,
                "think_style": "",
            },
        )
        event = FakeEvent()
        key = m.merge_key(event)
        window = m.MergeWindow(1.0, 30.0, first_message="hello")
        plugin.pending[key] = window

        calls = []

        async def fake_think(
            event, first_message, window, elapsed=0.0, remaining=0.0, style_context=""
        ):
            calls.append(window.unseen)
            if len(calls) == 1:
                return m.Verdict([], "", [], "wait", 3.0)
            return m.Verdict([], "", [], "reply", 0.0)

        plugin._think = fake_think

        async def donate_later():
            await asyncio.sleep(0.4)
            window.donate((2, 0, 1), "follow-up", [], [])

        async def scenario():
            donator = asyncio.create_task(donate_later())
            await plugin._hold_window(key, event, window, True)
            await donator

        asyncio.run(scenario())
        # First pass read the leader; the lazy slide kept the follow-up
        # unread; the final pass at the deadline read it and said reply.
        assert calls == [0, 1], calls
        assert window.close_reason == "quiet", window.close_reason
        assert window.unseen == 0 and len(window.thoughts) == 2
        assert key not in plugin.pending
    finally:
        m.THINK_SETTLE_SECONDS = settle
        m.THINK_REPLY_PAUSE = pause
    print("lazy loop OK")


class FakeProvider:
    """Capture real thought requests and delegate completion to a test callback."""

    provider_config = {"modalities": ["image"]}

    def __init__(self, callback=None):
        self.calls = []
        self.callback = callback

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.callback is not None:
            return await self.callback(len(self.calls), kwargs)
        return types.SimpleNamespace(completion_text="A useful thought <<reply>>")


class FakeContext:
    """Provide in-memory provider selection, history and runner configuration."""

    def __init__(self, provider=None, history=(), runner="local"):
        self.provider = provider or FakeProvider()
        self.history = list(history)
        self.runner = runner
        self.conversation_manager = self

    async def get_using_provider_async(self, **_kwargs):
        return self.provider

    async def get_curr_conversation_id(self, _umo):
        return "conversation"

    async def get_conversation(self, _umo, _conversation_id):
        return types.SimpleNamespace(history=json.dumps(self.history))

    def get_config(self, **_kwargs):
        return {"agent_runner": {"runner_type": self.runner}}


def request(prompt="hello", images=()):
    """Build the mutable request fields shared by merge-tail scenarios.

    Args:
        prompt: Leader text already prepared by the host.
        images: Leader image references already prepared by the host.

    Returns:
        An independent request double.
    """
    return types.SimpleNamespace(
        prompt=prompt, image_urls=list(images), extra_user_content_parts=[], func_tool=None
    )


async def test_real_think_unseen(m):
    """Keep donations made during a real provider call unread until a later pass.

    Args:
        m: Imported plugin module.
    """
    for outcome in ("success", "exception", "invalid"):
        first_started = asyncio.Event()
        finish_first = asyncio.Event()
        second_started = asyncio.Event()
        finish_second = asyncio.Event()

        async def complete(index, _kwargs):
            if index == 1:
                first_started.set()
                await finish_first.wait()
                if outcome == "exception":
                    raise RuntimeError("Simulated provider failure")
                raw = "" if outcome == "invalid" else "First thought <<reply>>"
                return types.SimpleNamespace(completion_text=raw)
            second_started.set()
            await finish_second.wait()
            return types.SimpleNamespace(completion_text="Second thought <<reply>>")

        provider = FakeProvider(complete)
        plugin = m.ReplyMergePlugin(
            FakeContext(provider),
            {"min_wait": 0.02, "max_wait": 0.03, "think_context": False},
        )
        event = FakeEvent()
        key = m.merge_key(event)
        window = m.MergeWindow(0.02, 2, first_message="leader")
        window.donate((1, 0, 1), "before provider", ["seed-image"], [])
        plugin.pending[key] = window
        leader = asyncio.create_task(plugin._hold_window(key, event, window, True))
        try:
            await asyncio.wait_for(first_started.wait(), 1)
            window.donate((2, 0, 2), "during provider", ["late-image"], [])
            finish_first.set()
            await asyncio.wait_for(second_started.wait(), 1)
            assert "during provider" not in provider.calls[0]["prompt"]
            assert "during provider" in provider.calls[1]["prompt"]
            assert provider.calls[0]["image_urls"] == ["seed-image"]
            if outcome == "success":
                assert window.unseen == 1
                assert window.think_images_seen == {"seed-image"}
                assert provider.calls[1]["image_urls"] == ["late-image"]
                assert "First thought" in provider.calls[1]["prompt"]
            else:
                assert window.unseen == 2
                assert window.think_images_seen == set()
                assert provider.calls[1]["image_urls"] == ["seed-image", "late-image"]
                assert window.thoughts[0].action == "fallback"
            finish_second.set()
            await asyncio.wait_for(leader, 1)
            assert window.unseen == 0
            assert window.think_images_seen == {"seed-image", "late-image"}
            assert len(provider.calls) == 2
            assert key not in plugin.pending
        finally:
            finish_first.set()
            finish_second.set()
            if not leader.done():
                leader.cancel()
            await asyncio.gather(leader, return_exceptions=True)
    print("real think donation and failure retries OK")


async def test_hard_cap_budget(m):
    """Bound context, queue and model awaits by the same window budget.

    Args:
        m: Imported plugin module.
    """
    for stage in ("style", "history", "queue", "model", "combined"):
        cancelled = asyncio.Event()
        model_entered = asyncio.Event()

        async def complete(_index, _kwargs):
            model_entered.set()
            try:
                await asyncio.sleep(0.14 if stage == "combined" else 2)
            finally:
                cancelled.set()
            return types.SimpleNamespace(completion_text="Done <<reply>>")

        plugin = m.ReplyMergePlugin(
            FakeContext(FakeProvider(complete)),
            {
                "min_wait": 0,
                "max_wait": 0.01,
                "think_context": stage == "history",
                "think_style": "Test voice",
            },
        )
        plugin._think_gate = asyncio.Semaphore(1)
        event = FakeEvent()
        key = m.merge_key(event)
        window = m.MergeWindow(0.01, 1, first_message="leader")
        budget = 0.2 if stage == "combined" else 0.08
        window.hard_cap = time.monotonic() + budget
        plugin.pending[key] = window

        async def slow_lookup(_event):
            try:
                await asyncio.sleep(2)
            finally:
                cancelled.set()
            return "Slow context"

        if stage == "style":
            plugin._style_context = slow_lookup
        elif stage == "history":
            plugin._history_block = slow_lookup
        queued = stage in ("queue", "combined")
        releaser = None
        if queued:
            await plugin._think_gate.acquire()
            if stage == "combined":

                async def release_gate():
                    await asyncio.sleep(0.14)
                    plugin._think_gate.release()

                releaser = asyncio.create_task(release_gate())
        started = time.monotonic()
        try:
            await asyncio.wait_for(plugin._hold_window(key, event, window, True), 1)
            elapsed = time.monotonic() - started
            assert elapsed < budget + 0.055, (stage, elapsed, budget)
            assert window.close_reason == "hard_cap", (stage, window.close_reason)
            assert key not in plugin.pending
            if stage == "queue":
                assert not model_entered.is_set()
                assert plugin._think_gate.locked()
                plugin._think_gate.release()
            else:
                assert cancelled.is_set(), stage
            if stage in ("style", "history"):
                assert not model_entered.is_set()
            elif stage in ("model", "combined"):
                assert model_entered.is_set()
            # Acquire and release after timeout to prove cancellation restored capacity.
            await asyncio.wait_for(plugin._think_gate.acquire(), 0.1)
            assert plugin._think_gate.locked()
            plugin._think_gate.release()
            assert not plugin._think_gate.locked()
        finally:
            if releaser is not None:
                await releaser
    print("shared hard-cap budget and gate cancellation OK")


async def test_history_cleanup(m):
    """Remove only genuine user thought blocks while keeping message bodies.

    Args:
        m: Imported plugin module.
    """
    plugin = m.ReplyMergePlugin(None, {})
    event = FakeEvent()
    window = m.MergeWindow(
        1, 10, first_message="current leader", leader_order=(5, 0, 0)
    )
    window.earlier = [m.BufferedMessage((1, 0, 0), ["- earlier body"], [])]
    window.donate((3, 0, 0), "before body", [], [])
    window.donate((6, 0, 0), "followup body", [], [])
    window.thoughts = [m.Verdict([], "", [], "reply", 0, "PRIVATE_NEW_THOUGHT")]
    window.waits = [0]
    event.set_extra("_reply_merge_window", window)
    req = request("current leader")
    await plugin.merge_tail(event, req)
    (draft,) = req.extra_user_content_parts
    assert draft.temporary
    assert f"[reply_merge:thought:{window.nonce}]\n" in draft.text
    assert f"\n[/reply_merge:thought:{window.nonce}]" in draft.text
    assert "PRIVATE_NEW_THOUGHT" not in req.prompt
    old_header = (
        "[以下是你在回复前对这位用户这句话的内心思考（校验码 deadbeef），帮助理解]\n"
        "PRIVATE_OLD_THOUGHT"
    )
    legacy = (
        "old leader\n\n"
        + old_header
        + "\n\n[以下是这位用户更早发来、但上一轮没能回复的内容（校验码 deadbeef），也请回应]\n"
        "- old earlier\n\n"
        "[以下是这位用户在最后这条消息之前先发出的内容（校验码 deadbeef），顺序在先]\n"
        "- old before\n\n"
        "[以下是这位用户在这条消息之后紧接着连发的后续内容（校验码 deadbeef），请合并]\n"
        "- old followup"
    )
    ordinary = "[以下是用户自己写的说明]\nordinary user body"
    assistant = old_header + "\nassistant continuation"
    malformed = "[以下是你在回复前对" + "内心思考（校验码 abcdef12）xxxxxxxxxx" * 4000
    plugin.context = FakeContext(
        history=[
            {"role": "user", "content": req.prompt},
            {"role": "user", "content": legacy},
            {"role": "user", "content": ordinary},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": malformed},
        ]
    )
    history = await plugin._history_block(event)
    lines = history.splitlines()
    assert len(lines) == 5, history
    assert "PRIVATE_NEW_THOUGHT" not in lines[0]
    for body in ("current leader", "earlier body", "before body", "followup body"):
        assert body in lines[0], history
    assert "reply_merge:thought:" not in lines[0]
    assert "PRIVATE_OLD_THOUGHT" not in lines[1]
    for body in ("old leader", "old earlier", "old before", "old followup"):
        assert body in lines[1], history
    assert "校验码" not in lines[1]
    assert "[以下是用户自己写的说明]" in lines[2]
    assert "ordinary user body" in lines[2]
    assert lines[3].startswith("你：")
    assert old_header.replace("\n", " / ") in lines[3]
    assert "assistant continuation" in lines[3]
    assert malformed[:80] in lines[4]
    print("bounded thought markers and legacy history cleanup OK")


async def test_image_message_ownership(m):
    """Keep images with whole selected messages and restore sending order.

    Args:
        m: Imported plugin module.
    """
    for forward in (True, False):
        for limit in (1, 2):
            plugin = m.ReplyMergePlugin(
                None, {"max_followups": limit, "forward_images": forward}
            )
            event = FakeEvent()
            window = m.MergeWindow(
                1,
                10,
                first_message="leader body",
                leader_order=(5, 0, 0),
                leader_images=["leader-image"],
            )
            window.earlier = [
                m.BufferedMessage((1, 0, 0), ["- earlier"], ["earlier-image"])
            ]
            window.donate((9, 0, 0), "latest pair", ["latest-a", "latest-b"], ["文件"])
            window.donate((2, 0, 0), "before", ["before-image"], [])
            window.donate((7, 0, 0), "old omitted", ["omitted-image"], [])
            window.donate((8, 0, 0), "", ["middle-image"], [])
            assert window.all_images() == [
                "earlier-image",
                "before-image",
                "leader-image",
                "omitted-image",
                "middle-image",
                "latest-a",
                "latest-b",
            ]
            event.set_extra("_reply_merge_window", window)
            req = request("leader body", ["leader-image"])
            await plugin.merge_tail(event, req)
            expected = ["leader-image"]
            if forward:
                expected = ["earlier-image", "before-image", "leader-image"]
                if limit == 2:
                    expected.append("middle-image")
                expected.extend(["latest-a", "latest-b"])
            assert req.image_urls == expected, (forward, limit, req.image_urls)
            assert "old omitted" not in req.prompt
            assert "latest pair" in req.prompt and "- [文件]" in req.prompt
            assert ("- [图片/表情包]" in req.prompt) is (limit == 2)
            assert (
                req.prompt.index("- earlier")
                < req.prompt.index("- before")
                < req.prompt.index("leader body")
            )
            assert (
                plugin.last_chain[m.merge_key(event)]["messages"][-1]
                == "latest pair\n- [文件]"
            )
            provider = FakeProvider()
            plugin.context = FakeContext(provider)
            plugin.config["think_context"] = False
            await plugin._think(event, window.first_message, window, 0, 1)
            assert provider.calls[0]["image_urls"] == (
                window.all_images()[: m.THINK_IMAGE_LIMIT]
                if forward
                else ["leader-image"]
            )

    # A late image sorts before pictures already consumed by an earlier pass.
    provider = FakeProvider()
    plugin = m.ReplyMergePlugin(FakeContext(provider), {"think_context": False})
    window = m.MergeWindow(1, 10, first_message="leader", leader_order=(5, 0, 0))
    window.donate((9, 0, 0), "later", ["late-a", "late-b"], [])
    await plugin._think(FakeEvent(), "leader", window, 0, 1)
    window.donate((2, 0, 0), "arrived out of order", ["early-a", "early-b"], [])
    await plugin._think(FakeEvent(), "leader", window, 0, 1)
    assert provider.calls[0]["image_urls"] == ["late-a", "late-b"]
    assert provider.calls[1]["image_urls"] == ["early-a", "early-b"]
    assert window.think_images_seen == {"late-a", "late-b", "early-a", "early-b"}
    print("whole-message image truncation and late-arrival ordering OK")


async def test_orphan_stale_move_order(m):
    """Move parked and stale bursts with image ownership exactly once.

    Args:
        m: Imported plugin module.
    """
    plugin = m.ReplyMergePlugin(None, {"enable_thinking": False})
    event = FakeEvent(text="new leader", images=["new-image"], order=10)
    key = m.merge_key(event)
    orphan = m.MergeWindow(
        1,
        10,
        first_message="orphan leader",
        leader_order=(2, 0, 0),
        leader_images=["orphan-image"],
    )
    orphan.donate((1, 0, 0), "orphan before", ["orphan-before-image"], [])
    orphan.donate((3, 0, 0), "orphan after", ["orphan-after-image"], [])
    stale = m.MergeWindow(
        1,
        10,
        first_message="stale leader",
        leader_order=(5, 0, 0),
        leader_images=["stale-image"],
    )
    stale.donate((4, 0, 0), "stale before", ["stale-before-image"], [])
    stale.donate((6, 0, 0), "stale after", ["stale-after-image"], [])
    recent = m.MergeWindow(1, 10, first_message="recent parked")
    expired = m.MergeWindow(1, 10, first_message="expired parked")
    now = time.monotonic()
    plugin.orphaned[key] = [
        (now - m.ORPHAN_GRACE_SECONDS - 1, orphan),
        (now, recent),
        (now - m.ORPHAN_MAX_AGE_SECONDS - 1, expired),
    ]
    plugin.inflight[key] = (stale, now)
    plugin._random_quiet = lambda: 0
    await plugin.hold_window(event)
    new = event.get_extra("_reply_merge_window")
    assert [message.order[0] for message in new.earlier] == [1, 2, 3, 4, 5, 6]
    assert new.all_images() == [
        "orphan-before-image",
        "orphan-image",
        "orphan-after-image",
        "stale-before-image",
        "stale-image",
        "stale-after-image",
        "new-image",
    ]
    assert orphan.rescued and stale.rescued
    assert orphan.take_messages() == stale.take_messages() == []
    assert plugin.stale[key] == {stale}
    assert [old for _ts, old in plugin.orphaned[key]] == [recent, new]
    req = request("new leader", ["new-image"])
    await plugin.merge_tail(event, req)
    assert req.image_urls == new.all_images()
    assert "recent parked" not in req.prompt and "expired parked" not in req.prompt
    for old in (orphan, stale):
        old_event = FakeEvent()
        old_event.set_extra("_reply_merge_window", old)
        old_req = request("superseded")
        await plugin.merge_tail(old_event, old_req)
        assert old_event.is_stopped()
        assert old_req.prompt == "superseded"
    print("orphan and stale message/image moves OK")


async def test_terminate_drains_pipeline(m):
    """Drain actual host tasks through queueing, prompt merge and response.

    Args:
        m: Imported plugin module.
    """
    for runner in ("local", "third-party"):
        provider_started = asyncio.Event()
        provider_finished = asyncio.Event()
        finish_provider = asyncio.Event()
        waiting_done = asyncio.Event()
        allow_response = asyncio.Event()
        response_started = asyncio.Event()
        pipeline_done = asyncio.Event()
        session_gate = asyncio.Lock()
        await session_gate.acquire()

        async def complete(_index, _kwargs):
            provider_started.set()
            try:
                await finish_provider.wait()
            finally:
                provider_finished.set()
            return types.SimpleNamespace(completion_text="Ready <<reply>>")

        plugin = m.ReplyMergePlugin(
            FakeContext(FakeProvider(complete), runner=runner),
            {
                "min_wait": 0.02,
                "max_wait": 0.03,
                "think_style": "Test voice",
                "think_context": False,
            },
        )
        event = FakeEvent(text="accepted leader", order=1)
        key = m.merge_key(event)
        req = request("accepted leader")
        resp = types.SimpleNamespace(completion_text="merged response")

        async def pipeline():
            if runner == "local":
                await plugin.hold_window(event)
                waiting_done.set()
                async with session_gate:
                    await plugin.merge_tail(event, req)
                    await plugin.finalize_request(event, req)
            else:
                await plugin.merge_tail(event, req)
                waiting_done.set()
                async with session_gate:
                    pass
            response_started.set()
            await allow_response.wait()
            await plugin.drop_stale_reply(event, resp)
            pipeline_done.set()

        leader = asyncio.create_task(pipeline())
        management = None
        try:
            await asyncio.wait_for(provider_started.wait(), 1)
            followup = FakeEvent(
                text="accepted followup", images=["followup-image"], order=2
            )
            await plugin.collect_followup(followup)
            assert followup.call_llm and not followup.is_stopped()
            window = plugin.pending[key]
            management = asyncio.create_task(plugin.terminate())
            await asyncio.sleep(0)
            assert plugin._closing
            late = FakeEvent(text="late must remain independent", order=3)
            await plugin.collect_followup(late)
            assert not late.is_stopped() and len(window.buffer) == 1
            assert not management.done()
            finish_provider.set()
            # Closing wakes the window, but must retain the host pipeline task.
            await asyncio.wait_for(waiting_done.wait(), 1)
            assert not management.done() and not pipeline_done.is_set()
            if runner == "local":
                assert event.get_extra("_reply_merge_window") is window
                assert key in plugin.orphaned
                assert req.prompt == "accepted leader"
            session_gate.release()
            await asyncio.wait_for(response_started.wait(), 1)
            assert "accepted followup" in req.prompt
            assert "late must remain independent" not in req.prompt
            assert req.image_urls == ["followup-image"]
            if runner == "local":
                assert plugin.inflight[key][0] is window
            else:
                assert key not in plugin.inflight
            assert not management.done()
            allow_response.set()
            await asyncio.wait_for(leader, 1)
            await asyncio.wait_for(management, 1)
            assert pipeline_done.is_set() and resp.completion_text == "merged response"
            assert not plugin.pending and not plugin.orphaned and not plugin.inflight
            assert not plugin.stale and not plugin.last_chain
            assert provider_finished.is_set()
        finally:
            if session_gate.locked():
                session_gate.release()
            allow_response.set()
            if not leader.done():
                leader.cancel()
            await asyncio.gather(leader, return_exceptions=True)
            if management is not None:
                await asyncio.gather(management, return_exceptions=True)
    print("management unload drains local and third-party pipelines OK")


async def test_leader_cancel_and_unload_guard(m):
    """Park cancelled leaders and reject unloads from them or their children.

    Args:
        m: Imported plugin module.
    """
    for stage in ("style", "model", "queue", "fallback"):
        entered = asyncio.Event()

        async def blocked(*_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

        plugin = m.ReplyMergePlugin(
            FakeContext(
                FakeProvider(blocked),
                runner="third-party" if stage == "fallback" else "local",
            ),
            {
                "min_wait": 0,
                "max_wait": 0.02,
                "think_style": "Test voice",
                "think_context": False,
            },
        )
        if stage == "style":
            plugin._style_context = blocked
        if stage == "queue":
            plugin._think_gate = asyncio.Semaphore(0)
        event = FakeEvent(text="cancelled leader", images=["leader-image"], order=1)
        key = m.merge_key(event)
        leader = asyncio.create_task(
            plugin.merge_tail(event, request())
            if stage == "fallback"
            else plugin.hold_window(event)
        )
        try:
            if stage == "queue":
                # Style resolution yields once before the real gate acquisition.
                for _ in range(20):
                    await asyncio.sleep(0.002)
                    if plugin._think_gate._waiters:
                        break
                assert plugin._think_gate._waiters
            else:
                await asyncio.wait_for(entered.wait(), 1)
            window = plugin.pending[key]
            followup = FakeEvent(
                text="cancelled followup", images=["followup-image"], order=2
            )
            await plugin.collect_followup(followup)
            leader.cancel()
            result = await asyncio.gather(leader, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
            assert event.is_stopped()
            assert key not in plugin.pending
            assert [old for _ts, old in plugin.orphaned[key]] == [window]
            assert window.first_message == "cancelled leader"
            assert window.buffer[0].lines == ["- cancelled followup"]
            assert window.all_images() == ["leader-image", "followup-image"]
            if stage != "queue":
                assert not plugin._think_gate.locked()
            else:
                plugin._think_gate.release()
                await asyncio.wait_for(plugin._think_gate.acquire(), 0.1)
                plugin._think_gate.release()
        finally:
            if not leader.done():
                leader.cancel()
            await asyncio.gather(leader, return_exceptions=True)

    for child in (False, True):
        plugin = m.ReplyMergePlugin(None, {"enable_thinking": False})
        plugin._random_quiet = lambda: 0
        event = FakeEvent()

        async def self_unload():
            await plugin.hold_window(event)
            req = request()
            await plugin.merge_tail(event, req)
            await plugin.finalize_request(event, req)
            if child:
                await asyncio.create_task(plugin.terminate())
            else:
                await plugin.terminate()

        leader = asyncio.create_task(self_unload())
        result = await asyncio.gather(leader, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError), (child, result)
        assert not plugin._closing
        assert m.merge_key(event) in plugin.inflight
    print("leader cancellation parking and self/child unload guards OK")


async def test_cancel_management_request(m):
    """Keep accepted host tasks alive when the management request is cancelled.

    Args:
        m: Imported plugin module.
    """
    plugin = m.ReplyMergePlugin(None, {"enable_thinking": False})
    plugin._random_quiet = lambda: 0
    queued = asyncio.Event()
    finish = asyncio.Event()
    event = FakeEvent(text="accepted before cancelled unload")
    req = request(event.get_message_str())

    async def pipeline():
        await plugin.hold_window(event)
        queued.set()
        await finish.wait()
        await plugin.merge_tail(event, req)
        await plugin.finalize_request(event, req)
        await plugin.drop_stale_reply(
            event, types.SimpleNamespace(completion_text="reply")
        )

    leader = asyncio.create_task(pipeline())
    management = None
    try:
        await asyncio.wait_for(queued.wait(), 1)
        management = asyncio.create_task(plugin.terminate())
        await asyncio.sleep(0)
        assert plugin._closing and not leader.done()
        management.cancel()
        result = await asyncio.gather(management, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert not plugin._closing and not leader.done()
        assert plugin._debounce_enabled(FakeEvent())
        finish.set()
        await asyncio.wait_for(leader, 1)
        assert not event.is_stopped()
        assert req.prompt == "accepted before cancelled unload"
        assert not plugin.inflight
        await plugin.terminate()
    finally:
        finish.set()
        if not leader.done():
            leader.cancel()
        await asyncio.gather(leader, return_exceptions=True)
        if management is not None:
            await asyncio.gather(management, return_exceptions=True)
    print("cancelled management unload preserves the accepted pipeline OK")


async def test_unanswered_handoff(m):
    """Rescue unanswered input once across unload and reload on the same host.

    Args:
        m: Imported plugin module.
    """
    for failure in ("cancelled", "provider-selection", "request-hook", "reload"):
        context = FakeContext()
        plugin = m.ReplyMergePlugin(context, {"enable_thinking": False})
        plugin._random_quiet = lambda: 0.02
        event = FakeEvent(
            text="unanswered leader", images=["old-leader-image"], order=5
        )
        key = m.merge_key(event)

        async def old_pipeline():
            await plugin.hold_window(event)
            if failure == "request-hook":
                await plugin.merge_tail(event, request("unanswered leader"))
                event.stop_event()
                return
            raise RuntimeError("Simulated host provider-selection failure")

        leader = asyncio.create_task(old_pipeline())
        await asyncio.sleep(0)
        assert key in plugin.pending
        for order, text, images in (
            (8, "old after", ["old-after-a", "old-after-b"]),
            (2, "old before", ["old-before-image"]),
        ):
            await plugin.collect_followup(
                FakeEvent(text=text, images=images, order=order)
            )
        if failure == "cancelled":
            leader.cancel()
        result = await asyncio.gather(leader, return_exceptions=True)
        if failure == "request-hook":
            assert event.is_stopped() and result == [None]
        else:
            assert isinstance(
                result[0],
                asyncio.CancelledError if failure == "cancelled" else RuntimeError,
            )
        assert len(plugin.orphaned[key]) == 1
        now = time.monotonic()
        old = plugin.orphaned[key][0][1]
        old_closed_at = (
            now if failure == "provider-selection" else now - m.ORPHAN_GRACE_SECONDS - 1
        )
        expired = m.MergeWindow(1, 10, first_message="expired handoff")
        plugin.orphaned[key] = [
            (old_closed_at, old),
            (now - m.ORPHAN_MAX_AGE_SECONDS - 1, expired),
        ]
        await plugin.terminate()
        current_module = load_plugin() if failure == "reload" else m
        replacement = current_module.ReplyMergePlugin(
            context, {"enable_thinking": False}
        )
        replacement._random_quiet = lambda: 0
        assert key in replacement.orphaned
        if failure == "provider-selection":
            assert (
                time.monotonic() - replacement.orphaned[key][0][0]
                >= m.ORPHAN_GRACE_SECONDS
            )
        else:
            assert replacement.orphaned[key][0][0] == old_closed_at
        assert not hasattr(context, "_reply_merge_handoff")
        assert not current_module.ReplyMergePlugin(context, {}).orphaned

        async def new_pipeline(current_event, current_req):
            await replacement.hold_window(current_event)
            await replacement.merge_tail(current_event, current_req)
            await replacement.finalize_request(current_event, current_req)
            await replacement.drop_stale_reply(
                current_event, types.SimpleNamespace(completion_text="rescued response")
            )

        new_event = FakeEvent(text="new leader", images=["new-leader-image"], order=10)
        req = request("new leader", ["new-leader-image"])
        await asyncio.create_task(new_pipeline(new_event, req))
        for text in ("old before", "unanswered leader", "old after", "new leader"):
            assert text in req.prompt, (failure, req.prompt)
        assert "expired handoff" not in req.prompt
        assert (
            req.prompt.index("old before")
            < req.prompt.index("unanswered leader")
            < req.prompt.index("old after")
        )
        assert req.image_urls == [
            "old-before-image",
            "old-leader-image",
            "old-after-a",
            "old-after-b",
            "new-leader-image",
        ]
        next_event = FakeEvent(text="next independent", order=11)
        next_req = request("next independent")
        await asyncio.create_task(new_pipeline(next_event, next_req))
        assert next_req.prompt == "next independent"
        assert next_req.image_urls == []
        assert not replacement.orphaned
        await replacement.terminate()
    print("cancelled and failed host requests survive reload and rescue once OK")


async def run_async_regressions(m):
    """Run scheduling regressions with short, restored timing constants.

    Args:
        m: Imported plugin module.
    """
    settle, pause = m.THINK_SETTLE_SECONDS, m.THINK_REPLY_PAUSE
    m.THINK_SETTLE_SECONDS = 0.001
    m.THINK_REPLY_PAUSE = 0.001
    try:
        await test_fresh_history_boundary(m)
        await test_peer_event_and_request_interop(m)
        await test_real_think_unseen(m)
        await test_hard_cap_budget(m)
        await test_history_cleanup(m)
        await test_image_message_ownership(m)
        # Host task context belongs to a pipeline and must not escape to later tests.
        for test in (
            test_orphan_stale_move_order,
            test_terminate_drains_pipeline,
            test_leader_cancel_and_unload_guard,
            test_cancel_management_request,
            test_unanswered_handoff,
        ):
            await asyncio.create_task(test(m))
    finally:
        m.THINK_SETTLE_SECONDS = settle
        m.THINK_REPLY_PAUSE = pause


async def test_peer_event_and_request_interop(m):
    context = FakeContext()
    plugin = m.ReplyMergePlugin(context, {})
    leader = FakeEvent(text="leader")
    key = m.merge_key(leader)
    window = m.MergeWindow(1, 10, first_message="leader")
    plugin.pending[key] = window

    for name, marker in (
        ("w1_event", {"is_user_message": False}),
        ("play_together_handled", True),
        ("provider_request", request("plugin prompt")),
        (
            "activated_handlers",
            [types.SimpleNamespace(event_filters=[type("RegexFilter", (), {})()])],
        ),
    ):
        peer = FakeEvent()
        peer.set_extra(name, marker)
        await plugin.collect_followup(peer)
        await plugin.hold_window(peer)
        assert not peer.call_llm and not peer.is_stopped() and not window.buffer
        assert peer.get_extra("_reply_merge_window") is None

    already_claimed = FakeEvent()
    already_claimed.should_call_llm(True)
    await plugin.collect_followup(already_claimed)
    assert not window.buffer
    followup = FakeEvent(text="sticker context", images=["sticker.png"])
    await plugin.collect_followup(followup)
    assert followup.call_llm and not followup.is_stopped()
    leader.set_extra("_reply_merge_window", window)
    req = request("leader")
    await plugin.merge_tail(leader, req)
    assert "sticker context" in req.prompt and req.image_urls == ["sticker.png"]

    # The final tool list can be supplied by a request hook after merge_tail.
    req.func_tool = ["w1_delegate_to_codex"]
    await plugin.finalize_request(leader, req)
    assert key not in plugin.inflight
    req.func_tool = None
    leader.set_extra("enable_streaming", True)
    await plugin.finalize_request(leader, req)
    assert key not in plugin.inflight
    leader.set_extra("enable_streaming", False)
    leader.set_extra("action_type", "live")
    await plugin.finalize_request(leader, req)
    assert key not in plugin.inflight
    leader.set_extra("action_type", None)
    await plugin.finalize_request(leader, req)
    assert plugin.inflight[key][0] is window
    await plugin.terminate()
    print("peer event routing, merged images and protected replies OK")


async def test_fresh_history_boundary(m):
    plugin = m.ReplyMergePlugin(FakeContext(), {})
    event = FakeEvent(umo="test:FriendMessage:alice", private=True)
    history = [{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "追问"}]
    now = time.time()
    moment = {"last_user": now - 40000, "last_reply": now - 39900}
    clocks = types.SimpleNamespace(sessions={event.unified_msg_origin: moment}, boundary=lambda *_: 2)
    peer = types.SimpleNamespace(config={}, closing=False, clocks=clocks, number=lambda *_: 6)
    plugin.context.get_registered_star = lambda _: types.SimpleNamespace(activated=True, star_cls=peer)
    assert plugin._fresh_history(event, types.SimpleNamespace(), history) == []
    assert moment["last_user"] == now - 40000  # A draft cannot reset the clock.
    moment["last_reply"] = now - 1
    fresh = history + [{"role": "user", "content": "今天想说的"}]
    assert plugin._fresh_history(event, types.SimpleNamespace(), fresh) == fresh[2:]
    plugin.context.get_registered_star = lambda _: None
    assert plugin._fresh_history(event, types.SimpleNamespace(updated_at=now - 40000), history) == []
    marked = "今天\n[reply_merge:thought:1234abcd]\n旧催问\n[/reply_merge:thought:1234abcd]"
    original = [{"role": "user", "content": marked}]
    assert m.without_saved_drafts(original) == [{"role": "user", "content": "今天"}]
    assert original[0]["content"] == marked
    await plugin.terminate()
    print("fresh history boundary, standalone fallback and saved draft cleanup OK")


def main() -> None:
    install_stubs()
    m = load_plugin()
    test_window_grouping(m)
    test_verdict_parsing(m)
    test_scope_gate(m)
    test_stale_reply_discard(m)
    test_stale_content_rides(m)
    test_lazy_loop(m)
    asyncio.run(run_async_regressions(m))
    print("ALL LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
