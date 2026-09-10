"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { conditionFormula, timeframeName } from "./formula-format.js";
import { isStrategyScanning } from "./run-status.js";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8010/api";

type MarketStatus = {
  market: string;
  state: string;
  quote_time: string | null;
  data_status: string;
  message: string | null;
  initialized: number;
  backfilled: number;
  total: number;
};

type Stock = {
  security_id: string;
  code: string;
  name: string;
  market: string;
  currency: string;
  latest_price: number | null;
  quote_time: string | null;
  signal_active: boolean;
  signal_reason: string | null;
  signal_stale: boolean;
  added_by?: string;
  quantity?: number;
  avg_cost?: number;
  unrealized_pnl?: number | null;
  unrealized_pnl_pct?: number | null;
  add_count?: number;
  last_add_price?: number | null;
  last_add_at?: string | null;
  first_add_at?: string | null;
  first_add_20d_low?: number | null;
  first_add_20d_low_date?: string | null;
  first_add_post_low_high?: number | null;
  first_add_rebound_pct?: number | null;
  first_add_rebound_confirmed?: boolean | null;
  add_events?: Array<{
    id: string;
    quantity: number;
    price: number;
    occurred_at: string;
    detail: Record<string, unknown>;
  }>;
};

type Rule = {
  id: string;
  scope: "universe" | "candidate" | "position";
  timeframe: "day" | "week" | "mixed";
  action: string;
  label: string;
  target_position_pct?: number | null;
  condition: Condition;
};

type Indicator = {
  name: string;
  field: string;
  params?: Record<string, number | string>;
  timeframe?: Rule["timeframe"];
};

type Condition = {
  op: string;
  conditions?: Condition[];
  left?: Indicator;
  right?: Indicator | number;
  comparator?: string;
  periods?: number;
  lower?: number;
  upper?: number;
  direction?: "high" | "low";
  timeframe?: Rule["timeframe"];
};

type Run = {
  id: string;
  strategy_id: string;
  strategy_name?: string;
  status: string;
  started_at: string;
  finished_at?: string | null;
  candidates_added: number;
  signals_active: number;
  securities_scanned: number;
  error?: string | null;
};

type Signal = {
  id: string;
  strategy_name: string;
  security_name: string;
  code: string;
  market: string;
  active: number;
  reason: string;
  created_at: string;
};

type Strategy = {
  id: string;
  name: string;
  description: string;
  active: boolean;
  current_version: number;
  dsl: { schema_version: 1; rules: Rule[] };
  explanation: string[];
  candidates: Stock[];
  positions: Stock[];
  latest_run: Run | null;
};

type StrategyDialogue = {
  id: string;
  strategy_id: string | null;
  strategy_name: string;
  original_description: string;
  normalized_text: string;
  compiled_dsl: Strategy["dsl"] | null;
  explanation: string[];
  validation_errors: string[];
  ai_status: "not_requested" | "requested" | "success" | "failed" | "disabled";
  ai_error: string | null;
  status: "clarifying" | "ready" | "activated";
  ready: boolean;
  messages: Array<{ id: string; role: "user" | "assistant"; content: string; created_at: string }>;
};

type Dashboard = {
  strategies: Strategy[];
  market_status: MarketStatus[];
  recent_runs: Run[];
  recent_signals: Signal[];
  trading: TradingDashboard;
  backtesting: BacktestingDashboard;
  training: TrainingDashboard;
};

type TradingIntent = {
  id: string;
  strategy_name: string;
  security_name: string;
  code: string;
  market: string;
  currency: string;
  action: string;
  side: "buy" | "sell";
  status: string;
  target_position_pct: number;
  desired_quantity: number | null;
  filled_quantity: number;
  limit_price: number | null;
  approval_deadline: string | null;
  blocked_reason: string | null;
  reason: string;
};

type TradingDashboard = {
  account: {
    mode: "paper" | "shadow" | "live";
    status: string;
    entry_paused: number;
    emergency_stop: number;
    pending_approvals: number;
    open_orders: number;
    health: { connected: boolean; adapter: string };
    balances: Array<{ currency: string; cash: number; available: number; frozen: number }>;
    allocations: Array<{ strategy_id: string; strategy_name: string; currency: string; capital_limit: number; enabled: number }>;
  };
  intents: TradingIntent[];
  fills: Array<{ id: string; security_name: string; code: string; market: string; currency: string; strategy_name: string; side: string; quantity: number; price: number; filled_at: string }>;
};

type BacktestSummary = {
  securities: number;
  trading_days: number;
  signals: number;
  trades: number;
  closed_trades: number;
  win_rate_pct: number | null;
  total_return_pct: number;
  annualized_return_pct: number;
  max_drawdown_pct: number;
  sharpe: number;
  cny_return_pct: number;
  hkd_return_pct: number;
  total_fees: number;
  signal_analysis: Record<string, { count: number; average_return_pct: number | null; win_rate_pct: number | null }>;
  survivorship_warning: string;
};

type BacktestRun = {
  id: string;
  strategy_id: string;
  strategy_name: string;
  status: string;
  start_date: string;
  end_date: string;
  created_at: string;
  error: string | null;
  summary: BacktestSummary;
  config: Record<string, number | string | boolean | null>;
};

type BacktestDetail = BacktestRun & {
  equity_curve: Array<{ trade_date: string; cny_equity: number; hkd_equity: number; normalized_equity: number; drawdown_pct: number }>;
  trades: Array<{ id: string; security_name: string; code: string; market: string; currency: string; side: string; quantity: number; price: number; fee: number; trade_date: string; action: string; reason: string; realized_pnl: number | null }>;
  signals: Array<{ id: string; security_name: string; code: string; market: string; signal_date: string; action: string; reason: string }>;
};

type BacktestingDashboard = {
  runs: BacktestRun[];
  latest: BacktestDetail | null;
};

type TrainingMarketQuality = {
  start_date: string | null;
  end_date: string | null;
  years: number;
  bars: number;
  universe: number;
  securities_with_bars: number;
  coverage_pct: number;
  history_gate: boolean;
  coverage_gate: boolean;
  freshness_target_date: string | null;
  freshness_universe: number;
  fresh_securities: number;
  freshness_coverage_pct: number;
  freshness_gate: boolean;
  status_coverage_pct: number;
  inactive_universe: number;
  inactive_with_bars: number;
  inactive_coverage_pct: number;
  point_in_time_status: boolean;
  ready_for_training: boolean;
  blocking_reasons: string[];
  warnings: string[];
};

type TrainingCampaign = {
  id: string;
  track: string;
  market: "A" | "HK";
  style: "left" | "right";
  status: string;
  trigger_type: string;
  budget: number;
  completed_trials: number;
  progress: number;
  summary: Record<string, unknown> & { best_score?: number; gates?: { passed?: boolean; reasons?: string[] } };
  error: string | null;
  created_at: string;
};

type TrainingChampion = {
  track: string;
  market: "A" | "HK";
  status: string;
  validation_tier?: "strict_champion" | "paper_observation" | "research_only";
  strategy_id?: string | null;
  campaign_id?: string;
  trial_id?: string;
  params: Record<string, number | string | boolean>;
  gates: Record<string, boolean | number | string[]> & { passed?: boolean; reasons?: string[] };
  performance?: {
    holdout?: { total_return_pct?: number; annual_return_pct?: number; max_drawdown_pct?: number; sharpe?: number; trades?: number };
    dsr_probability?: number | null;
    annual_excess_return_pct?: number | null;
  };
  data_ready: boolean;
  data_blockers: string[];
  strategy_blockers: string[];
  evaluation_status: "pending_data" | "evaluated";
};

type TrainingDashboard = {
  data_quality: { minimum_years: number; ready_for_promotion: boolean; markets: Record<"A" | "HK", TrainingMarketQuality> };
  campaigns: TrainingCampaign[];
  champions: TrainingChampion[];
};

type SearchStock = {
  id: string;
  market: string;
  code: string;
  name: string;
  currency: string;
  latest_price: number | null;
};

type Modal =
  | { type: "new-strategy" }
  | { type: "edit-strategy"; strategy: Strategy }
  | { type: "add-stock"; strategy: Strategy; destination: "candidate" | "position" }
  | { type: "promote"; strategy: Strategy; stock: Stock }
  | { type: "increase-position"; strategy: Strategy; stock: Stock }
  | null;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : body.detail?.message || message;
    } catch {}
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

function shortTime(value?: string | null) {
  if (!value) return "尚未同步";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(value));
}

function price(value: number | null | undefined, currency = "CNY") {
  if (value == null) return "—";
  const symbol = currency === "HKD" ? "HK$" : "¥";
  return `${symbol}${value.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function marketName(market: string) {
  return { SH: "沪市", SZ: "深市", HK: "港股" }[market] || market;
}

function ScopeBadge({ scope }: { scope: Rule["scope"] }) {
  const map = {
    universe: ["全市场", "scope-universe"],
    candidate: ["待定池", "scope-candidate"],
    position: ["持仓池", "scope-position"],
  } as const;
  return <span className={`scope-badge ${map[scope][1]}`}>{map[scope][0]}</span>;
}

function actionName(rule: Rule) {
  const names: Record<string, string> = {
    add_candidate: "加入待定",
    highlight_candidate: "优选关注",
    highlight_position: "持仓提醒",
    signal_buy: "建议建仓",
    signal_add: "建议加仓",
    signal_hold: "继续持有",
    signal_reduce: "建议减仓",
    signal_exit: "建议清仓",
    signal_stop: "止损清仓",
  };
  const target = rule.target_position_pct == null ? "" : ` ${rule.target_position_pct}%`;
  return `${names[rule.action] || rule.action}${target}`;
}

function RuleFormula({ rule, index }: { rule: Rule; index: number }) {
  const fallback = rule.timeframe === "mixed" ? "day" : rule.timeframe;
  return (
    <details className="rule-detail">
      <summary className="rule-row">
        <span className="rule-index">{String(index + 1).padStart(2, "0")}</span>
        <ScopeBadge scope={rule.scope} />
        <span className="timeframe">{timeframeName(rule.timeframe)}</span>
        <span className="rule-label" title={rule.label}>{rule.label}</span>
        <span className="rule-arrow">→</span>
        <span className="rule-action">{actionName(rule)}</span>
        <span className="rule-expand">查看公式⌄</span>
      </summary>
      <div className="rule-formula-detail">
        <small>实际判断公式</small>
        <code>{conditionFormula(rule.condition, fallback)}</code>
        <div className="rule-formula-meta">
          <span>对象：{rule.scope === "universe" ? "全市场" : rule.scope === "candidate" ? "待定池" : "持仓池"}</span>
          <span>动作：{actionName(rule)}</span>
          {rule.target_position_pct != null && <span>目标仓位：{rule.target_position_pct}%</span>}
        </div>
        <details className="rule-json">
          <summary>查看原始 DSL JSON</summary>
          <pre>{JSON.stringify(rule, null, 2)}</pre>
        </details>
      </div>
    </details>
  );
}

function StrategyCard({
  strategy,
  onRefresh,
  onModal,
  notify,
}: {
  strategy: Strategy;
  onRefresh: () => Promise<void>;
  onModal: (modal: Modal) => void;
  notify: (message: string, error?: boolean) => void;
}) {
  const [submitting, setSubmitting] = useState(false);
  const running = isStrategyScanning(submitting, strategy.latest_run?.status);

  const action = async (fn: () => Promise<unknown>, success: string) => {
    try {
      await fn();
      notify(success);
      await onRefresh();
    } catch (error) {
      notify(error instanceof Error ? error.message : "操作失败", true);
    }
  };

  const execute = async () => {
    setSubmitting(true);
    try {
      await action(() => request(`/strategies/${strategy.id}/execute`, { method: "POST" }), "策略执行完成");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <article className={`strategy-card ${strategy.active ? "" : "strategy-paused"}`}>
      <header className="strategy-header">
        <div>
          <div className="strategy-eyebrow">
            <span className={`live-dot ${strategy.active ? "" : "paused"}`} />
            {strategy.active ? "自动跟踪中" : "已暂停"} · V{strategy.current_version}
          </div>
          <h2>{strategy.name}</h2>
        </div>
        <div className="strategy-actions">
          <button className="icon-button" onClick={() => onModal({ type: "edit-strategy", strategy })} title="编辑策略" aria-label={`编辑${strategy.name}`}>✎</button>
          <button
            className="icon-button"
            onClick={() => action(
              () => request(`/strategies/${strategy.id}`, { method: "PATCH", body: JSON.stringify({ active: !strategy.active }) }),
              strategy.active ? "策略已暂停" : "策略已启用",
            )}
            title={strategy.active ? "暂停" : "启用"}
          >{strategy.active ? "Ⅱ" : "▶"}</button>
          <button
            className="icon-button danger"
            onClick={() => {
              if (window.confirm(`确定归档策略“${strategy.name}”吗？运行和交易历史会保留。`)) {
                action(() => request(`/strategies/${strategy.id}`, { method: "DELETE" }), "策略已归档");
              }
            }}
            title="删除策略"
          >×</button>
        </div>
      </header>

      <div className="strategy-summary">
        <p>{strategy.description}</p>
        <button className="run-button" onClick={execute} disabled={running || !strategy.active}>
          <span>{running ? "◌" : "▷"}</span>{running ? "正在扫描" : "立即执行"}
        </button>
      </div>

      <section className="formula-panel">
        <div className="section-title-row">
          <h3>执行公式</h3>
          <span>{strategy.dsl.rules.length} 条规则</span>
        </div>
        <div className="rule-list">
          {strategy.dsl.rules.map((rule, index) => <RuleFormula rule={rule} index={index} key={rule.id} />)}
        </div>
      </section>

      <div className="lists-grid">
        <StockList
          title="待定股票"
          count={strategy.candidates.length}
          stocks={strategy.candidates}
          type="candidate"
          onAdd={() => onModal({ type: "add-stock", strategy, destination: "candidate" })}
          onDelete={(stock) => action(
            () => request(`/strategies/${strategy.id}/candidates/${stock.security_id}`, { method: "DELETE" }),
            `${stock.name}已移出待定`,
          )}
          onMove={(stock) => onModal({ type: "promote", strategy, stock })}
        />
        <StockList
          title="模拟持仓"
          count={strategy.positions.length}
          stocks={strategy.positions}
          type="position"
          onAdd={() => onModal({ type: "add-stock", strategy, destination: "position" })}
          onDelete={(stock) => action(
            () => request(`/strategies/${strategy.id}/positions/${stock.security_id}`, { method: "DELETE" }),
            `${stock.name}持仓已删除`,
          )}
          onMove={(stock) => action(
            () => request(`/strategies/${strategy.id}/positions/${stock.security_id}/return`, { method: "POST" }),
            `${stock.name}已放回待定`,
          )}
          onIncrease={(stock) => onModal({ type: "increase-position", strategy, stock })}
        />
      </div>

      <footer className="strategy-footer">
        <span>最近执行 {shortTime(strategy.latest_run?.started_at)}</span>
        <span>{strategy.latest_run ? `扫描 ${strategy.latest_run.securities_scanned} · 新增 ${strategy.latest_run.candidates_added} · 信号 ${strategy.latest_run.signals_active}` : "等待首次执行"}</span>
      </footer>
    </article>
  );
}

function StockList({
  title,
  count,
  stocks,
  type,
  onAdd,
  onDelete,
  onMove,
  onIncrease,
}: {
  title: string;
  count: number;
  stocks: Stock[];
  type: "candidate" | "position";
  onAdd: () => void;
  onDelete: (stock: Stock) => void;
  onMove: (stock: Stock) => void;
  onIncrease?: (stock: Stock) => void;
}) {
  const initialRows = 80;
  const rowBatch = 200;
  const [visibleCount, setVisibleCount] = useState(initialRows);
  const visibleStocks = stocks.slice(0, visibleCount);
  const remaining = Math.max(0, stocks.length - visibleStocks.length);

  return (
    <section className="stock-list">
      <div className="section-title-row list-title">
        <h3>{title}<em>{count}</em></h3>
        <button className="text-button" onClick={onAdd}>＋ 添加</button>
      </div>
      <div className="stock-table">
        {stocks.length === 0 && <div className="empty-state">列表为空，可手动添加股票</div>}
        {visibleStocks.map((stock) => (
          <div className={`stock-row ${stock.signal_active ? "has-signal" : ""}`} key={stock.security_id}>
            <div className="stock-identity">
              <span className={`market-tag market-${stock.market.toLowerCase()}`}>{stock.market}</span>
              <div>
                <strong>{stock.name}</strong>
                <small>{stock.code}</small>
              </div>
            </div>
            <div className="stock-metrics">
              <strong>{price(stock.latest_price, stock.currency)}</strong>
              {type === "position" && stock.unrealized_pnl_pct != null ? (
                <small className={stock.unrealized_pnl_pct >= 0 ? "positive" : "negative"}>
                  {stock.unrealized_pnl_pct >= 0 ? "+" : ""}{stock.unrealized_pnl_pct.toFixed(2)}%
                </small>
              ) : <small>{stock.added_by === "auto" ? "策略加入" : "手动加入"}</small>}
            </div>
            {type === "position" && (
              <div className="position-meta">
                <span>{stock.quantity?.toLocaleString()} 股</span>
                <small>成本 {price(stock.avg_cost, stock.currency)}</small>
              </div>
            )}
            <div className="row-actions">
              {type === "position" && onIncrease && <button className="increase" onClick={() => onIncrease(stock)}>＋ 记录补仓</button>}
              <button onClick={() => onMove(stock)}>{type === "candidate" ? "转持仓" : "放回待定"}</button>
              <button className="remove" onClick={() => onDelete(stock)} aria-label={`删除${stock.name}`}>×</button>
            </div>
            {type === "position" && Boolean(stock.add_count) && (
              <details className="position-add-state">
                <summary>
                  已补仓 {stock.add_count} 次 · 前次 {price(stock.last_add_price, stock.currency)} ·
                  <span className={stock.first_add_rebound_confirmed ? "positive" : "negative"}>
                    {stock.first_add_rebound_confirmed ? " 首次补仓快照已确认5%反弹" : " 首次补仓快照未确认5%反弹"}
                  </span>
                </summary>
                <div className="position-snapshot">
                  <span>首次补仓</span><strong>{shortTime(stock.first_add_at)}</strong>
                  <span>20日最低</span><strong>{price(stock.first_add_20d_low, stock.currency)} · {stock.first_add_20d_low_date || "—"}</strong>
                  <span>最低点后最高</span><strong>{price(stock.first_add_post_low_high, stock.currency)}</strong>
                  <span>反弹幅度</span><strong>{stock.first_add_rebound_pct == null ? "—" : `${stock.first_add_rebound_pct.toFixed(2)}%`}</strong>
                </div>
                <p>固定公式：最低点出现之后的日K最高价 ≥ 20日最低价 × 1.05。相同最低价按最后一次计算，最低点当日不计入之后。</p>
                <div className="position-add-history">
                  {(stock.add_events || []).map((event, index) => (
                    <span key={event.id}>#{index + 1} {shortTime(event.occurred_at)} · {event.quantity.toLocaleString()}股 @ {price(event.price, stock.currency)}</span>
                  ))}
                </div>
              </details>
            )}
            {stock.signal_active && (
              <div className={`signal-strip ${stock.signal_stale ? "stale" : ""}`}>
                <span>↗</span>{stock.signal_reason}{stock.signal_stale ? " · 数据待更新" : ""}
              </div>
            )}
          </div>
        ))}
        {remaining > 0 && (
          <button
            className="stock-load-more"
            onClick={() => setVisibleCount((current) => Math.min(stocks.length, current + rowBatch))}
          >
            继续显示 {Math.min(rowBatch, remaining)} 只
            <small>尚有 {remaining.toLocaleString()} 只未渲染</small>
          </button>
        )}
      </div>
    </section>
  );
}

function Dialog({ children, onClose, wide = false }: { children: React.ReactNode; onClose: () => void; wide?: boolean }) {
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className={`modal ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="关闭">×</button>
        {children}
      </div>
    </div>
  );
}

function StrategyDialogueModal({ strategy, close, refresh, notify }: { strategy?: Strategy; close: () => void; refresh: () => Promise<void>; notify: (value: string, error?: boolean) => void }) {
  const [name, setName] = useState(strategy?.name || "");
  const [description, setDescription] = useState(strategy?.description || "");
  const [dialogue, setDialogue] = useState<StrategyDialogue | null>(null);
  const [reply, setReply] = useState("");
  const [working, setWorking] = useState(false);
  const trimmedName = name.trim();
  const descriptionChanged = Boolean(strategy && description.trim() !== strategy.description.trim());
  const nameChanged = Boolean(strategy && trimmedName !== strategy.name);
  const renameOnly = Boolean(strategy && nameChanged && !descriptionChanged);
  const unchanged = Boolean(strategy && !nameChanged && !descriptionChanged);

  const start = async (event: FormEvent) => {
    event.preventDefault();
    setWorking(true);
    try {
      if (strategy && !descriptionChanged) {
        if (nameChanged) {
          await request(`/strategies/${strategy.id}`, {
            method: "PATCH",
            body: JSON.stringify({ name: trimmedName }),
          });
          notify("策略名称已更新，不需要调用AI");
          await refresh();
          close();
        }
        return;
      }
      setDialogue(await request<StrategyDialogue>("/strategy-dialogues", {
        method: "POST",
        body: JSON.stringify({ name: trimmedName, description, strategy_id: strategy?.id || null }),
      }));
    } catch (error) { notify(error instanceof Error ? error.message : "无法开始策略对话", true); }
    finally { setWorking(false); }
  };

  const send = async (event: FormEvent) => {
    event.preventDefault();
    if (!dialogue || !reply.trim()) return;
    setWorking(true);
    try {
      setDialogue(await request<StrategyDialogue>(`/strategy-dialogues/${dialogue.id}/messages`, {
        method: "POST",
        body: JSON.stringify({ content: reply.trim() }),
      }));
      setReply("");
    } catch (error) { notify(error instanceof Error ? error.message : "对话更新失败", true); }
    finally { setWorking(false); }
  };

  const activate = async () => {
    if (!dialogue?.ready) return;
    setWorking(true);
    try {
      await request(`/strategy-dialogues/${dialogue.id}/activate`, { method: "POST" });
      if (strategy && trimmedName !== strategy.name) {
        await request(`/strategies/${strategy.id}`, {
          method: "PATCH",
          body: JSON.stringify({ name: trimmedName }),
        });
      }
      notify(strategy ? `策略 V${strategy.current_version + 1} 已确认启用` : "新策略已创建并启用");
      await refresh();
      close();
    } catch (error) { notify(error instanceof Error ? error.message : "启用失败", true); }
    finally { setWorking(false); }
  };

  const retryAI = async () => {
    if (!dialogue) return;
    setWorking(true);
    try {
      setDialogue(await request<StrategyDialogue>(`/strategy-dialogues/${dialogue.id}/retry`, { method: "POST" }));
    } catch (error) { notify(error instanceof Error ? error.message : "AI重试失败", true); }
    finally { setWorking(false); }
  };

  return (
    <Dialog onClose={close} wide>
      <div className="modal-heading">
        <span>{strategy && !descriptionChanged ? "EDIT STRATEGY" : `AI STRATEGY DIALOGUE${strategy ? ` · V${strategy.current_version + 1}` : ""}`}</span>
        <h2>{strategy ? `编辑 ${strategy.name}` : "通过对话创建策略"}</h2>
        <p>{strategy && !descriptionChanged
          ? "只修改策略名称会直接保存，不调用AI，也不会创建新的策略版本；修改原始自然语言后才会进入AI规则对话。"
          : "AI负责澄清含糊条件；只有规则化自然语言能被确定性编译并通过校验后，策略才可启用。"}</p>
      </div>
      {!dialogue ? (
        <form onSubmit={start} className="form-stack">
          <label>策略名称<input autoFocus value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：周线趋势回踩右侧交易" required /></label>
          <label>原始自然语言{strategy ? "（修改后调用AI）" : ""}<textarea value={description} onChange={(event) => setDescription(event.target.value)} placeholder="可以使用附近、明显、适量等自然表达，AI会继续追问具体定义。" rows={10} required /></label>
          <button className="primary-button" disabled={working || unchanged}>
            {working ? (renameOnly ? "正在保存名称…" : "AI正在分析…") : unchanged ? "没有需要保存的修改" : renameOnly ? "保存名称" : "开始对话并整理规则 →"}
          </button>
        </form>
      ) : (
        <div className="dialogue-grid">
          <section className="dialogue-chat">
            <div className="preview-header"><span>策略对话</span><em>{dialogue.status === "clarifying" ? "待确认" : dialogue.status === "ready" ? "可编译" : "已启用"}</em></div>
            <div className={`ai-call-status ${working ? "requested" : dialogue.ai_status}`} title={dialogue.ai_error || ""}>
              <i />{working ? "正在请求AI，临时故障会自动重试…" : dialogue.ai_status === "success" ? "本轮AI已响应" : dialogue.ai_status === "failed" ? "AI请求失败 · 等待重试" : dialogue.ai_status === "disabled" ? "AI未配置 · 无法继续" : "等待发送消息"}
            </div>
            {dialogue.ai_status === "failed" && !working && (
              <div className="ai-retry-panel">
                <strong>AI连续重试后仍未响应</strong>
                <p>本轮已停止，不会使用本地校验生成回复或DSL。你可以保持当前对话并重新请求。</p>
                <button type="button" onClick={retryAI}>重试本轮AI请求</button>
              </div>
            )}
            <div className="chat-messages">
              {dialogue.messages.map((message) => (
                <div key={message.id} className={`chat-message ${message.role}`}>
                  <small>{message.role === "user" ? "你" : "策略助手"}</small>
                  <p>{message.content}</p>
                </div>
              ))}
              {working && <div className="chat-thinking">AI正在核对规则…</div>}
            </div>
            <form onSubmit={send} className="chat-reply">
              <textarea value={reply} onChange={(event) => setReply(event.target.value)} rows={3} placeholder="回答问题，或要求修改某个阈值……" />
              <button className="secondary-button" disabled={working || !reply.trim()}>发送并更新规则</button>
            </form>
          </section>
          <section className="controlled-preview">
            <div className="preview-header"><span>规则化自然语言</span><em>{dialogue.ready ? "确定性编译通过" : "存在未确认项"}</em></div>
            {dialogue.normalized_text ? <pre>{dialogue.normalized_text}</pre> : <div className="preview-empty">回答左侧问题后生成<br /><small>未确认前不会创建DSL</small></div>}
            {!!dialogue.validation_errors.length && <ul className="validation-errors">{dialogue.validation_errors.map((error) => <li key={error}>{error}</li>)}</ul>}
            {dialogue.compiled_dsl && (
              <details className="compiled-rules" open>
                <summary>已编译规则 · {dialogue.compiled_dsl.rules.length}条</summary>
                <div className="preview-rules">{dialogue.compiled_dsl.rules.map((rule, index) => <div key={rule.id}><b>{index + 1}</b><ScopeBadge scope={rule.scope} /><span>{timeframeName(rule.timeframe)} · {rule.label} → {actionName(rule)}</span></div>)}</div>
                <details><summary>查看DSL JSON</summary><pre>{JSON.stringify(dialogue.compiled_dsl, null, 2)}</pre></details>
              </details>
            )}
            <button className="primary-button" onClick={activate} disabled={working || !dialogue.ready}>{dialogue.ready ? (strategy ? `确认并启用 V${strategy.current_version + 1}` : "确认并创建策略") : "请先完成规则确认"}</button>
          </section>
        </div>
      )}
    </Dialog>
  );
}

function StockModal({ modal, close, refresh, notify }: { modal: Extract<Modal, { type: "add-stock" | "promote" }>; close: () => void; refresh: () => Promise<void>; notify: (value: string, error?: boolean) => void }) {
  const isPromote = modal.type === "promote";
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchStock[]>([]);
  const [selected, setSelected] = useState<SearchStock | Stock | null>(isPromote ? modal.stock : null);
  const [quantity, setQuantity] = useState("100");
  const [cost, setCost] = useState(isPromote ? String(modal.stock.latest_price || "") : "");
  const destination = isPromote ? "position" : modal.destination;
  const strategy = modal.strategy;

  useEffect(() => {
    if (isPromote || query.trim().length < 1) return;
    const handle = window.setTimeout(() => {
      request<SearchStock[]>(`/securities?query=${encodeURIComponent(query)}`).then(setResults).catch(() => setResults([]));
    }, 250);
    return () => window.clearTimeout(handle);
  }, [query, isPromote]);

  const save = async () => {
    if (!selected) return;
    try {
      if (destination === "candidate") {
        await request(`/strategies/${strategy.id}/candidates`, { method: "POST", body: JSON.stringify({ security_id: "id" in selected ? selected.id : selected.security_id }) });
      } else {
        const securityId = "id" in selected ? selected.id : selected.security_id;
        const path = isPromote ? `/strategies/${strategy.id}/candidates/${securityId}/promote` : `/strategies/${strategy.id}/positions`;
        await request(path, { method: "POST", body: JSON.stringify({ security_id: securityId, quantity: Number(quantity), avg_cost: Number(cost) }) });
      }
      notify(destination === "candidate" ? "股票已加入待定" : "模拟持仓已保存");
      await refresh(); close();
    } catch (error) { notify(error instanceof Error ? error.message : "保存失败", true); }
  };

  return (
    <Dialog onClose={close}>
      <div className="modal-heading"><span>{destination === "candidate" ? "CANDIDATE" : "SIMULATED POSITION"}</span><h2>{isPromote ? `将 ${modal.stock.name} 转入持仓` : destination === "candidate" ? "添加待定股票" : "添加模拟持仓"}</h2><p>目标策略：{strategy.name}</p></div>
      {!isPromote && <label className="search-field">搜索股票<input autoFocus value={query} onChange={(e) => { setQuery(e.target.value); setSelected(null); }} placeholder="输入代码或名称" /></label>}
      {!isPromote && results.length > 0 && <div className="search-results">{results.map((stock) => <button key={stock.id} className={selected && "id" in selected && selected.id === stock.id ? "selected" : ""} onClick={() => { setSelected(stock); setCost(String(stock.latest_price || "")); }}><span className={`market-tag market-${stock.market.toLowerCase()}`}>{stock.market}</span><strong>{stock.name}</strong><small>{stock.code}</small><em>{price(stock.latest_price, stock.currency)}</em></button>)}</div>}
      {selected && <div className="selected-stock"><span>已选择</span><strong>{selected.name}</strong><small>{selected.code} · {marketName(selected.market)}</small></div>}
      {destination === "position" && selected && <div className="form-row"><label>持仓数量<input type="number" min="0.0001" step="any" value={quantity} onChange={(e) => setQuantity(e.target.value)} /></label><label>平均成本<input type="number" min="0.0001" step="any" value={cost} onChange={(e) => setCost(e.target.value)} /></label></div>}
      <button className="primary-button" onClick={save} disabled={!selected || (destination === "position" && (!Number(quantity) || !Number(cost)))}>确认添加</button>
    </Dialog>
  );
}

function PositionIncreaseModal({ modal, close, refresh, notify }: { modal: Extract<Modal, { type: "increase-position" }>; close: () => void; refresh: () => Promise<void>; notify: (value: string, error?: boolean) => void }) {
  const [quantity, setQuantity] = useState("100");
  const [tradePrice, setTradePrice] = useState(String(modal.stock.latest_price || ""));
  const [occurredAt, setOccurredAt] = useState("");
  const [working, setWorking] = useState(false);

  const save = async () => {
    setWorking(true);
    try {
      await request(`/strategies/${modal.strategy.id}/positions/${modal.stock.security_id}/adds`, {
        method: "POST",
        body: JSON.stringify({
          quantity: Number(quantity),
          price: Number(tradePrice),
          occurred_at: occurredAt || null,
        }),
      });
      notify(`${modal.stock.name}补仓已记录，持仓成本和策略状态已更新`);
      await refresh();
      close();
    } catch (error) {
      notify(error instanceof Error ? error.message : "补仓记录失败", true);
    } finally {
      setWorking(false);
    }
  };

  return (
    <Dialog onClose={close}>
      <div className="modal-heading">
        <span>POSITION ADD</span>
        <h2>记录 {modal.stock.name} 补仓</h2>
        <p>首次补仓会冻结此前20个已完成交易日的低点与其后的反弹快照；后续补仓只更新前次补仓价、次数、数量和加权成本。</p>
      </div>
      <div className="selected-stock">
        <span>当前持仓</span>
        <strong>{modal.stock.name}</strong>
        <small>{modal.stock.quantity?.toLocaleString()}股 · 成本 {price(modal.stock.avg_cost, modal.stock.currency)}</small>
      </div>
      <div className="form-row">
        <label>本次补仓数量<input autoFocus type="number" min="0.0001" step="any" value={quantity} onChange={(event) => setQuantity(event.target.value)} /></label>
        <label>实际成交价<input type="number" min="0.0001" step="any" value={tradePrice} onChange={(event) => setTradePrice(event.target.value)} /></label>
      </div>
      <label>成交时间（留空表示现在）<input type="datetime-local" value={occurredAt} onChange={(event) => setOccurredAt(event.target.value)} /></label>
      {!modal.stock.add_count && (
        <div className="snapshot-notice">
          <strong>首次补仓快照</strong>
          <p>最低点出现之后的最高价必须达到最低价的1.05倍才确认反弹。若不足20个已完成日K，本次操作会停止并提示先补齐真实行情。</p>
        </div>
      )}
      <button className="primary-button" onClick={save} disabled={working || !Number(quantity) || !Number(tradePrice)}>
        {working ? "正在核对历史行情…" : "确认记录补仓"}
      </button>
    </Dialog>
  );
}

function TradingPanel({ data, strategies, refresh, notify }: {
  data: TradingDashboard;
  strategies: Strategy[];
  refresh: () => Promise<void>;
  notify: (message: string, error?: boolean) => void;
}) {
  const [strategyId, setStrategyId] = useState(strategies[0]?.id || "");
  const [currency, setCurrency] = useState<"CNY" | "HKD">("CNY");
  const [capital, setCapital] = useState("200000");
  const [working, setWorking] = useState("");
  const terminal = ["filled", "canceled", "expired", "blocked", "completed_noop"];
  const activeIntents = data.intents.filter((intent) => !terminal.includes(intent.status));

  const mutate = async (key: string, path: string, init: RequestInit, message: string) => {
    setWorking(key);
    try {
      await request(path, init);
      notify(message);
      await refresh();
    } catch (actionError) {
      notify(actionError instanceof Error ? actionError.message : "交易操作失败", true);
    } finally {
      setWorking("");
    }
  };

  return (
    <section className="trading-panel">
      <div className="trading-heading">
        <div><span>AUTO EXECUTION</span><h2>自动交易控制台</h2><p>买入需在次日 09:20 前批准；卖出与硬止损自动执行。当前启用本地模拟券商。</p></div>
        <div className="trading-state">
          <span className={`account-mode mode-${data.account.mode}`}>{data.account.mode === "paper" ? "模拟盘" : data.account.mode === "shadow" ? "影子盘" : "实盘"}</span>
          <strong>{data.account.health.connected ? "券商连接正常" : "券商连接中断"}</strong>
          <small>{data.account.health.adapter} · {data.account.entry_paused ? "买入已暂停" : "允许买入"}</small>
        </div>
      </div>

      <div className="trading-metrics">
        {data.account.balances.map((balance) => <div key={balance.currency}><span>{balance.currency} 可用资金</span><strong>{price(balance.available, balance.currency)}</strong><small>现金 {price(balance.cash, balance.currency)}</small></div>)}
        <div><span>待批准买单</span><strong>{data.account.pending_approvals}</strong><small>09:20 截止</small></div>
        <div><span>待执行订单</span><strong>{data.account.open_orders}</strong><small>每 15 秒处理</small></div>
      </div>

      <div className="trading-tools">
        <div className="allocation-form">
          <strong>策略资金配额</strong>
          <select value={strategyId} onChange={(event) => setStrategyId(event.target.value)}>{strategies.map((strategy) => <option key={strategy.id} value={strategy.id}>{strategy.name}</option>)}</select>
          <select value={currency} onChange={(event) => setCurrency(event.target.value as "CNY" | "HKD")}><option value="CNY">CNY</option><option value="HKD">HKD</option></select>
          <input aria-label="资金配额" type="number" min="1" value={capital} onChange={(event) => setCapital(event.target.value)} />
          <button disabled={!strategyId || !Number(capital) || !!working} onClick={() => mutate("allocation", `/trading/allocations/${strategyId}`, { method: "PUT", body: JSON.stringify({ currency, capital_limit: Number(capital), enabled: true }) }, "策略资金配额已启用")}>启用配额</button>
        </div>
        <div className="control-buttons">
          <button disabled={!!working} onClick={() => mutate("pause", "/trading/controls", { method: "PATCH", body: JSON.stringify({ entry_paused: !data.account.entry_paused }) }, data.account.entry_paused ? "买入已恢复" : "买入已暂停")}>{data.account.entry_paused ? "恢复买入" : "暂停买入"}</button>
          <button disabled={!!working} onClick={() => mutate("reconcile", "/trading/reconcile", { method: "POST" }, "账实对账已完成")}>账实对账</button>
          <button className={data.account.emergency_stop ? "resume" : "emergency"} disabled={!!working} onClick={() => mutate("emergency", "/trading/controls", { method: "PATCH", body: JSON.stringify({ emergency_stop: !data.account.emergency_stop }) }, data.account.emergency_stop ? "紧急停止已解除" : "已紧急停止并撤销活动订单")}>{data.account.emergency_stop ? "解除紧急停止" : "紧急停止"}</button>
        </div>
      </div>

      <div className="intent-table">
        <div className="intent-row intent-labels"><span>标的 / 策略</span><span>方向</span><span>目标 / 数量</span><span>状态</span><span>说明</span><span>操作</span></div>
        {activeIntents.length ? activeIntents.map((intent) => <div className="intent-row" key={intent.id}>
          <strong>{intent.security_name} <small>{intent.market}.{intent.code}<br />{intent.strategy_name}</small></strong>
          <span className={intent.side === "buy" ? "trade-buy" : "trade-sell"}>{intent.side === "buy" ? "买入" : intent.action === "signal_stop" ? "硬止损" : "卖出"}</span>
          <span>{intent.target_position_pct}%<small>{intent.desired_quantity ? `${intent.desired_quantity} 股` : "待计算"}</small></span>
          <span className={`intent-status status-${intent.status}`}>{intent.status === "pending_approval" ? "待批准" : intent.status === "approved" ? "待执行" : intent.status === "submitted" ? "已报单" : intent.status}</span>
          <span className="intent-reason">{intent.blocked_reason || intent.reason}</span>
          <span className="intent-actions">{intent.status === "pending_approval" ? <><button disabled={!!working} onClick={() => mutate(intent.id, `/trading/intents/${intent.id}/approve`, { method: "POST" }, "买单已批准")}>批准</button><button disabled={!!working} onClick={() => mutate(intent.id, `/trading/intents/${intent.id}/reject`, { method: "POST" }, "买单已拒绝")}>拒绝</button></> : <button disabled={!!working || intent.action === "signal_stop"} onClick={() => mutate(intent.id, `/trading/intents/${intent.id}/cancel`, { method: "POST" }, "订单已撤销")}>撤销</button>}</span>
        </div>) : <div className="trading-empty">当前没有等待批准或执行的订单。</div>}
      </div>
    </section>
  );
}

function BacktestPanel({ data, strategies, refresh, notify }: {
  data: BacktestingDashboard;
  strategies: Strategy[];
  refresh: () => Promise<void>;
  notify: (message: string, error?: boolean) => void;
}) {
  const today = new Date();
  const yearAgo = new Date(today); yearAgo.setFullYear(today.getFullYear() - 1);
  const [strategyId, setStrategyId] = useState(strategies[0]?.id || "");
  const [startDate, setStartDate] = useState(yearAgo.toISOString().slice(0, 10));
  const [endDate, setEndDate] = useState(today.toISOString().slice(0, 10));
  const [defaultPosition, setDefaultPosition] = useState("10");
  const [stopLoss, setStopLoss] = useState("3");
  const [maxHolding, setMaxHolding] = useState("60");
  const [running, setRunning] = useState(false);
  const latest = data.latest;
  const activeRun = data.runs.find((item) => item.status === "running");
  const summary = latest?.summary;
  const curve = latest?.equity_curve || [];
  const chartPoints = curve.map((point, index) => {
    const x = curve.length <= 1 ? 0 : index / (curve.length - 1) * 760;
    const values = curve.map((item) => item.normalized_equity);
    const min = Math.min(...values, 99);
    const max = Math.max(...values, 101);
    const y = 150 - (point.normalized_equity - min) / Math.max(0.0001, max - min) * 140;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  const run = async () => {
    setRunning(true);
    try {
      await request<BacktestDetail>("/backtests", {
        method: "POST",
        body: JSON.stringify({
          strategy_id: strategyId, start_date: startDate, end_date: endDate,
          initial_cny: 1_000_000, initial_hkd: 1_000_000,
          default_position_pct: Number(defaultPosition), max_security_pct: 10,
          commission_pct: 0.03, min_commission: 5, sell_tax_pct: 0.05, slippage_pct: 0.1,
          stop_loss_pct: stopLoss ? Number(stopLoss) : null,
          max_holding_days: maxHolding ? Number(maxHolding) : null,
          force_close: true, max_securities: 2000,
        }),
      });
      notify("回测已进入后台运行，完成后页面会自动更新");
      await refresh();
    } catch (runError) {
      notify(runError instanceof Error ? runError.message : "回测失败", true);
    } finally {
      setRunning(false);
    }
  };

  const pct = (value: number | null | undefined) => value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
  return (
    <section className="backtest-panel">
      <div className="backtest-heading">
        <div><span>HISTORICAL REPLAY</span><h2>策略回测</h2><p>T日收盘判定、下一交易日开盘成交；已计入滑点、费用、目标仓位、硬止损和最长持有期。</p></div>
        <button disabled={running || !!activeRun || !strategyId || !startDate || !endDate} onClick={run}>{running || activeRun ? "逐日重放中…" : "运行回测"}</button>
      </div>
      {activeRun && <div className="backtest-running"><i />正在后台回测 {activeRun.strategy_name}（{activeRun.start_date} → {activeRun.end_date}），完成后自动刷新结果。</div>}
      <div className="backtest-form">
        <label>策略<select value={strategyId} onChange={(event) => setStrategyId(event.target.value)}>{strategies.map((strategy) => <option key={strategy.id} value={strategy.id}>{strategy.name}</option>)}</select></label>
        <label>开始日期<input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} /></label>
        <label>结束日期<input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} /></label>
        <label>缺省目标仓位<input type="number" min="1" max="100" value={defaultPosition} onChange={(event) => setDefaultPosition(event.target.value)} /><small>%</small></label>
        <label>建仓低点止损<input type="number" min="0.1" max="50" value={stopLoss} onChange={(event) => setStopLoss(event.target.value)} /><small>%</small></label>
        <label>最长持有<input type="number" min="1" max="2500" value={maxHolding} onChange={(event) => setMaxHolding(event.target.value)} /><small>交易日</small></label>
      </div>
      {latest && summary ? <>
        <div className="backtest-result-head"><div><strong>{latest.strategy_name}</strong><span>{latest.start_date} → {latest.end_date}</span></div><small>{summary.securities}只股票 · {summary.trading_days}个交易日 · {summary.signals}个信号</small></div>
        <div className="backtest-metrics">
          <div><span>组合收益</span><strong className={summary.total_return_pct >= 0 ? "positive" : "negative"}>{pct(summary.total_return_pct)}</strong><small>年化 {pct(summary.annualized_return_pct)}</small></div>
          <div><span>最大回撤</span><strong className="negative">{pct(summary.max_drawdown_pct)}</strong><small>夏普 {summary.sharpe.toFixed(2)}</small></div>
          <div><span>平仓胜率</span><strong>{pct(summary.win_rate_pct)}</strong><small>{summary.closed_trades}笔平仓</small></div>
          <div><span>CNY / HKD</span><strong>{pct(summary.cny_return_pct)}</strong><small>{pct(summary.hkd_return_pct)}</small></div>
          <div><span>成交与费用</span><strong>{summary.trades}</strong><small>{summary.total_fees.toFixed(2)}</small></div>
        </div>
        <div className="backtest-body">
          <div className="equity-chart"><div><strong>归一化净值</strong><span>起点 100</span></div>{chartPoints ? <svg viewBox="0 0 760 160" preserveAspectRatio="none" role="img" aria-label="回测净值曲线"><line x1="0" y1="150" x2="760" y2="150" /><polyline points={chartPoints} /></svg> : <p>暂无净值数据</p>}</div>
          <div className="signal-quality"><strong>买入信号后续表现</strong>{Object.entries(summary.signal_analysis).map(([horizon, item]) => <div key={horizon}><span>{horizon}日</span><b>{pct(item.average_return_pct)}</b><small>胜率 {pct(item.win_rate_pct)} · {item.count}次</small></div>)}</div>
        </div>
        <div className="backtest-trades"><div className="backtest-trade-row labels"><span>日期</span><span>股票</span><span>方向</span><span>数量 / 价格</span><span>损益</span><span>原因</span></div>{latest.trades.slice(-20).reverse().map((trade) => <div className="backtest-trade-row" key={trade.id}><span>{trade.trade_date}</span><strong>{trade.security_name}<small>{trade.market}.{trade.code}</small></strong><span className={trade.side === "buy" ? "trade-buy" : "trade-sell"}>{trade.side === "buy" ? "买入" : "卖出"}</span><span>{trade.quantity.toLocaleString()} / {price(trade.price, trade.currency)}</span><span className={(trade.realized_pnl || 0) >= 0 ? "positive" : "negative"}>{trade.realized_pnl == null ? "—" : price(trade.realized_pnl, trade.currency)}</span><span>{trade.reason}</span></div>)}</div>
        <p className="backtest-warning">{summary.survivorship_warning} 费用参数为可配置的研究假设，不代表真实券商结算。</p>
      </> : <div className="backtest-empty">选择策略和时间范围运行第一次回测；结果会持久保存。</div>}
    </section>
  );
}

function TrainingPanel({ data, refresh, notify }: {
  data: TrainingDashboard;
  refresh: () => Promise<void>;
  notify: (message: string, error?: boolean) => void;
}) {
  const [track, setTrack] = useState("all");
  const [working, setWorking] = useState("");
  const active = data.campaigns.find((campaign) => ["queued", "running"].includes(campaign.status));
  const trackNames: Record<string, string> = { "left-A": "左侧·A股", "right-A": "右侧·A股", "left-HK": "左侧·港股", "right-HK": "右侧·港股" };
  const gateNames: Record<string, string> = {
    data_history: "历史长度达到10年", data_coverage: "股票K线覆盖达到95%",
    data_freshness: "最新复权日K覆盖达到95%",
    point_in_time_status: "历史ST、停牌和退市状态完整", max_drawdown: "最大回撤不超过15%",
    positive_return: "留出集收益为正", sharpe: "留出集夏普比率达到0.8", profitable_folds: "3个开发区间全部盈利",
    trade_count: "开发集成交至少100笔且每段至少15笔", cost_stress: "额外成本压力测试盈利", pbo: "旧版PBO近似门槛（已停用）",
    deflated_sharpe: "开发集校正后夏普可信度达到95%", excess_return: "留出集跑赢同仓位市场基准",
    parameter_stability: "参数邻域表现稳定",
  };
  const parameterNames: Record<string, string> = {
    rsi_max: "RSI上限", rsi_min: "RSI下限", boll_b_max: "BOLL %B上限", low_window: "低点窗口",
    volume_ratio_max: "确认量比上限", volume_ratio_min: "放量比例下限", divergence_window: "背离窗口",
    confirm_return_max: "确认日涨幅上限", signal_return_max: "突破日涨幅上限", stop_pct: "止损百分比", exit_rsi: "退出RSI",
    max_hold: "最长持有日", breakout_window: "突破窗口", channel_exit_window: "通道退出窗口", adx_min: "ADX下限", stop_atr: "初始止损ATR倍数",
    trail_atr: "移动止损ATR倍数", use_boll: "启用BOLL", use_divergence: "启用背离",
    trend_floor_ratio: "价格/MA200下限", trend_slope_floor: "MA120趋势下限", adx_max: "ADX上限",
    trend_slope_days: "长期均线上升窗口", atr_pct_max: "ATR波动率上限", require_weekly: "要求周线共振",
  };
  const campaignNames: Record<string, string> = {
    queued: "等待执行", running: "训练中", research_only: "仅限研究", candidate_ready: "待批准",
    canceled: "已取消", failed: "失败", data_invalidated: "数据作废",
  };
  const valueLabel = (value: number | string | boolean) => typeof value === "number" ? Number(value.toFixed(4)).toString() : value === true ? "是" : value === false ? "否" : String(value);
  const tracks = track === "all" ? Object.keys(trackNames) : [track];
  const trackMarket = (value: string): "A" | "HK" => value.endsWith("-HK") ? "HK" : "A";
  const formalBlockedTracks = tracks.filter((value) => !data.data_quality.markets[trackMarket(value)].ready_for_training);
  const formalTrainingReady = formalBlockedTracks.length === 0;
  const mutate = async (key: string, path: string, init: RequestInit, message: string) => {
    setWorking(key);
    try { await request(path, init); notify(message); await refresh(); }
    catch (actionError) { notify(actionError instanceof Error ? actionError.message : "训练操作失败", true); }
    finally { setWorking(""); }
  };
  const start = (budget: number, trigger: "smoke" | "manual") => mutate(
    "start", "/training/campaigns",
    { method: "POST", body: JSON.stringify({ tracks, budget, trigger_type: trigger }) },
    trigger === "smoke" ? "快速训练已进入后台队列" : "完整训练已进入后台队列",
  );
  return <section className="training-panel">
    <div className="training-heading">
      <div><span>RULE STRATEGY LAB</span><h2>策略训练中心</h2><p>先完成数据建设，再进行多轮正式训练；快速训练仅验证流程，不作为晋级结论。</p></div>
      <div className="training-actions"><select value={track} onChange={(event) => setTrack(event.target.value)}><option value="all">全部四条轨道</option>{Object.entries(trackNames).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><button disabled={!!active || !!working} onClick={() => start(12, "smoke")}>快速检查</button><button title={formalTrainingReady ? "启动完整训练" : "所选市场的数据门槛尚未满足"} disabled={!!active || !!working || !formalTrainingReady} onClick={() => start(200, "manual")}>完整训练</button></div>
    </div>
    <div className={`training-phase ${data.data_quality.ready_for_promotion ? "phase-ready" : "phase-building"}`}><strong>{data.data_quality.ready_for_promotion ? "数据已就绪" : "数据建设中"}</strong><span>{data.data_quality.ready_for_promotion ? "四条轨道可进入正式多轮训练。" : "后台正在补齐历史K线、退市样本及点时状态；达到门槛后自动放行对应市场的正式训练。"}</span></div>
    {active && <div className="training-running"><i /><strong>{trackNames[active.track]}</strong><span>{active.status === "queued" ? "等待执行" : `已完成 ${active.completed_trials}/${active.budget}`}</span><div><b style={{ width: `${active.progress}%` }} /></div><button disabled={!!working} onClick={() => mutate("cancel", `/training/campaigns/${active.id}/cancel`, { method: "POST" }, "已请求安全取消训练")}>取消</button></div>}
    <div className="quality-grid">{(["A", "HK"] as const).map((market) => { const quality = data.data_quality.markets[market]; return <div key={market} className={quality.ready_for_training ? "quality-ok" : "quality-warning"}><div><span>{market === "A" ? "A股数据" : "港股数据"}</span><strong>{quality.ready_for_training ? "已就绪" : `${quality.years.toFixed(1)}年`}</strong></div><p>{quality.start_date || "—"} → {quality.end_date || "—"}</p><div className="quality-stats"><span>{quality.securities_with_bars}/{quality.universe}只</span><span>当前股票覆盖 {quality.coverage_pct.toFixed(1)}%</span><span>最新复权覆盖 {quality.freshness_coverage_pct.toFixed(1)}%（截至 {quality.freshness_target_date || "—"}）</span><span>{quality.bars.toLocaleString()}根K线</span><span>历史退市覆盖 {quality.inactive_with_bars}/{quality.inactive_universe}</span>{market === "A" && <span>ST/停牌日状态 {quality.status_coverage_pct.toFixed(1)}%</span>}</div><ul>{quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>; })}</div>
    <div className="champion-grid">{data.champions.map((champion) => {
      const displayStatus = !champion.data_ready ? "等待数据" : champion.status === "not_trained" ? "等待首轮训练" : champion.status === "research_only" ? champion.validation_tier === "paper_observation" ? champion.strategy_id ? "观察扫描中" : "可模拟观察" : champion.strategy_id ? "研究扫描中" : "策略待优化" : champion.status === "candidate_ready" ? "待批准" : "模拟盘观察";
      const strategyBlockers = champion.strategy_blockers || [];
      const canApprove = champion.data_ready && champion.gates.passed && champion.status === "candidate_ready";
      const canPublishResearch = champion.data_ready && !champion.gates.passed && champion.status === "research_only";
      const holdout = champion.performance?.holdout || {};
      const metric = (value: number | null | undefined, digits = 2) => typeof value === "number" ? value.toFixed(digits) : "—";
      return <div className="champion-card" key={champion.track}>
        <div className="champion-title"><div><span>{trackNames[champion.track]}</span><strong>{displayStatus}</strong></div><em className={`champion-status ${champion.data_ready ? champion.status : "data-building"} ${champion.validation_tier || ""}`}>{!champion.data_ready ? "数据建设" : champion.gates.passed ? "门槛通过" : champion.validation_tier === "paper_observation" ? "观察级" : "策略未达标"}</em></div>
        {Object.keys(champion.params).length ? <div className="champion-params">{Object.entries(champion.params).slice(0, 8).map(([key, value]) => <span key={key}><small>{parameterNames[key] || key}</small><b>{valueLabel(value)}</b></span>)}</div> : <p className="champion-empty">数据就绪后运行正式训练，显示冠军参数与样本外表现。</p>}
        {champion.performance && <div className="champion-performance"><span><small>留出集收益</small><b>{metric(holdout.total_return_pct)}%</b></span><span><small>留出集夏普</small><b>{metric(holdout.sharpe, 3)}</b></span><span><small>最大回撤</small><b>{metric(holdout.max_drawdown_pct)}%</b></span><span><small>去偏可信度</small><b>{metric(typeof champion.performance.dsr_probability === "number" ? champion.performance.dsr_probability * 100 : null, 1)}%</b></span></div>}
        {!champion.data_ready ? <div className="failed-gates data-pending">待补齐：{champion.data_blockers.map((reason) => gateNames[reason] || reason).join("、")}。现有策略结果仅用于流程检查，暂不判定优劣。</div> : strategyBlockers.length ? <div className="failed-gates">策略未通过：{strategyBlockers.map((reason) => gateNames[reason] || reason).join("、")}</div> : <div className="failed-gates passed-gates">数据和策略门槛均已满足。</div>}
        {canPublishResearch ? <button disabled={!!working} title="只接入选股和持仓提示，不配置资金，不生成交易订单" onClick={() => mutate(champion.track, `/training/champions/${champion.track}/publish-research`, { method: "POST" }, champion.strategy_id ? "扫描策略已更新" : "训练结果已接入扫描，不会生成交易订单")}>{champion.strategy_id ? champion.validation_tier === "paper_observation" ? "更新观察扫描" : "更新研究扫描" : champion.validation_tier === "paper_observation" ? "接入观察扫描" : "接入研究扫描"}</button> : <button disabled={!canApprove || !!working} onClick={() => mutate(champion.track, `/training/champions/${champion.track}/approve`, { method: "POST" }, "冠军已批准进入模拟盘观察")}>批准进入模拟盘</button>}
      </div>;
    })}</div>
    <div className="campaign-history"><div className="campaign-row labels"><span>轨道</span><span>状态</span><span>试验</span><span>最佳评分</span><span>触发方式</span><span>时间</span></div>{data.campaigns.slice(0, 12).map((campaign) => <div className="campaign-row" key={campaign.id}><strong>{trackNames[campaign.track]}</strong><span className={`campaign-status ${campaign.status}`}>{campaignNames[campaign.status] || campaign.status}</span><span>{campaign.completed_trials}/{campaign.budget}</span><span>{campaign.summary.best_score == null ? "—" : campaign.summary.best_score.toFixed(3)}</span><span>{campaign.trigger_type === "smoke" ? "快速检查" : campaign.trigger_type === "scheduled" ? "定时训练" : "手动完整训练"}</span><span>{shortTime(campaign.created_at)}</span></div>)}</div>
    <p className="training-note">后台持续补数；每日17:30检查数据就绪状态并在首次达标时启动正式训练，此后每周六02:00继续一轮挑战者迭代。未达标市场只允许运行12组快速流程检查。</p>
  </section>;
}

export function DashboardApp() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [error, setError] = useState("");
  const [modal, setModal] = useState<Modal>(null);
  const [toast, setToast] = useState<{ message: string; error: boolean } | null>(null);
  const [tab, setTab] = useState<"runs" | "signals">("runs");

  const refresh = useCallback(async () => {
    try {
      setDashboard(await request<Dashboard>("/dashboard"));
      setError("");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "无法连接本地服务");
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const events = new EventSource(`${API}/events`);
    const onHeartbeat = () => void refresh();
    events.addEventListener("heartbeat", onHeartbeat);
    const fallback = window.setInterval(refresh, 45_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(fallback);
      events.removeEventListener("heartbeat", onHeartbeat);
      events.close();
    };
  }, [refresh]);

  const notify = (message: string, isError = false) => {
    setToast({ message, error: isError });
    window.setTimeout(() => setToast(null), 3200);
  };

  const totals = useMemo(() => {
    const strategies = dashboard?.strategies || [];
    return {
      candidates: strategies.reduce((sum, strategy) => sum + strategy.candidates.length, 0),
      positions: strategies.reduce((sum, strategy) => sum + strategy.positions.length, 0),
      signals: strategies.reduce((sum, strategy) => sum + strategy.candidates.filter((stock) => stock.signal_active).length + strategy.positions.filter((stock) => stock.signal_active).length, 0),
    };
  }, [dashboard]);

  if (!dashboard && !error) return <div className="app-loading"><div className="loading-mark">量</div><p>正在载入策略工作台</p></div>;

  if (!dashboard && error) return (
    <main className="offline-page"><div className="offline-card"><span>LOCAL SERVICE OFFLINE</span><h1>量策前端已就绪</h1><p>本地数据服务尚未启动。启动后此页面会自动连接，并载入策略、持仓与执行记录。</p><code>./scripts/start-local.sh</code><button onClick={refresh}>重新连接</button><small>{error}</small></div></main>
  );

  return (
    <main className="dashboard-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark">量</div><div><strong>量策</strong><span>LIANGHUA · STRATEGY DESK</span></div></div>
        <div className="topbar-center"><span className="system-status"><i />系统运行中</span><span>每 15 分钟扫描</span><span>本地研究模式</span></div>
        <button className="new-strategy" onClick={() => setModal({ type: "new-strategy" })}>＋ 新建策略</button>
      </header>

      <section className="overview">
        <div className="overview-intro"><span>MARKET INTELLIGENCE</span><h1>策略工作台</h1><p>筛选全市场机会，跟踪每一次信号。</p></div>
        <div className="metric-card accent"><span>活跃信号</span><strong>{String(totals.signals).padStart(2, "0")}</strong><small>最近一次成功扫描</small></div>
        <div className="metric-card"><span>待定股票</span><strong>{String(totals.candidates).padStart(2, "0")}</strong><small>跨 {dashboard?.strategies.length} 个策略</small></div>
        <div className="metric-card"><span>模拟持仓</span><strong>{String(totals.positions).padStart(2, "0")}</strong><small>CNY / HKD 分币种核算</small></div>
      </section>

      <section className="market-strip">
        <div className="strip-label">市场状态</div>
        {dashboard?.market_status.map((status) => (
          <div className="market-status" key={status.market}>
            <span className={`market-pill ${status.state}`}>{status.market}</span>
            <div><strong>{marketName(status.market)}</strong><small>{status.state === "open" ? "交易中" : status.state === "lunch" ? "午间休市" : "已收盘"} · {shortTime(status.quote_time)}</small></div>
            <em>{status.data_status === "demo" ? "演示源" : "公开行情"}</em>
          </div>
        ))}
        <div className="data-progress"><span>历史数据</span><div><i style={{ width: `${Math.max(6, Math.min(100, ((dashboard?.market_status.reduce((s, m) => s + m.backfilled, 0) || 0) / (dashboard?.market_status.reduce((s, m) => s + m.total, 0) || 1)) * 100))}%` }} /></div><strong>{dashboard?.market_status.reduce((sum, item) => sum + item.backfilled, 0).toLocaleString()} / {dashboard?.market_status.reduce((sum, item) => sum + item.total, 0).toLocaleString()}</strong></div>
      </section>

      <section className="workspace-header"><div><span>STRATEGIES</span><h2>交易策略</h2></div><p>每个策略独立维护待定池、模拟持仓和信号版本</p></section>
      <section className="strategy-grid">
        {dashboard?.strategies.map((strategy) => <StrategyCard key={strategy.id} strategy={strategy} onRefresh={refresh} onModal={setModal} notify={notify} />)}
        <button className="add-card" onClick={() => setModal({ type: "new-strategy" })}><span>＋</span><strong>添加策略区域</strong><small>创建独立筛选与跟踪逻辑</small></button>
      </section>

      {dashboard?.trading && <TradingPanel data={dashboard.trading} strategies={dashboard.strategies} refresh={refresh} notify={notify} />}

      {dashboard?.backtesting && <BacktestPanel data={dashboard.backtesting} strategies={dashboard.strategies} refresh={refresh} notify={notify} />}

      {dashboard?.training && <TrainingPanel data={dashboard.training} refresh={refresh} notify={notify} />}

      <section className="activity-panel">
        <div className="activity-header"><div><span>ACTIVITY</span><h2>运行与信号历史</h2></div><div className="tabs"><button className={tab === "runs" ? "active" : ""} onClick={() => setTab("runs")}>执行记录</button><button className={tab === "signals" ? "active" : ""} onClick={() => setTab("signals")}>信号历史</button></div></div>
        {tab === "runs" ? <div className="activity-table"><div className="activity-row activity-labels"><span>策略</span><span>状态</span><span>开始时间</span><span>扫描 / 新增 / 信号</span></div>{dashboard?.recent_runs.length ? dashboard.recent_runs.map((run) => <div className="activity-row" key={run.id}><strong>{run.strategy_name}</strong><span className={`run-status ${run.status}`}>{run.status === "success" ? "成功" : run.status === "running" ? "执行中" : "失败"}</span><span>{shortTime(run.started_at)}</span><span>{run.securities_scanned} / {run.candidates_added} / {run.signals_active}</span></div>) : <div className="activity-empty">手动执行策略或等待定时扫描后，这里会显示运行记录。</div>}</div>
          : <div className="activity-table"><div className="activity-row signal-labels"><span>股票</span><span>策略</span><span>变化</span><span>原因</span><span>时间</span></div>{dashboard?.recent_signals.length ? dashboard.recent_signals.map((signal) => <div className="activity-row signal-labels" key={signal.id}><strong>{signal.security_name} <small>{signal.market}.{signal.code}</small></strong><span>{signal.strategy_name}</span><span className={signal.active ? "positive" : "muted"}>{signal.active ? "信号出现" : "信号消失"}</span><span>{signal.reason}</span><span>{shortTime(signal.created_at)}</span></div>) : <div className="activity-empty">信号发生变化后，这里会保留完整历史。</div>}</div>}
      </section>

      <footer className="disclaimer">本工具仅用于个人研究与模拟，不构成任何投资建议。免费行情可能延迟或中断，请以交易所及持牌数据源为准。</footer>

      {modal?.type === "new-strategy" && <StrategyDialogueModal close={() => setModal(null)} refresh={refresh} notify={notify} />}
      {modal?.type === "edit-strategy" && <StrategyDialogueModal strategy={modal.strategy} close={() => setModal(null)} refresh={refresh} notify={notify} />}
      {(modal?.type === "add-stock" || modal?.type === "promote") && <StockModal modal={modal} close={() => setModal(null)} refresh={refresh} notify={notify} />}
      {modal?.type === "increase-position" && <PositionIncreaseModal modal={modal} close={() => setModal(null)} refresh={refresh} notify={notify} />}
      {toast && <div className={`toast ${toast.error ? "toast-error" : ""}`}><span>{toast.error ? "!" : "✓"}</span>{toast.message}</div>}
    </main>
  );
}
