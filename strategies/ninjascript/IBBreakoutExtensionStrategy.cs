// =============================================================================
// IB Breakout Extension Strategy  (NQ / ES / YM)  -  NinjaTrader 8 / NinjaScript
// -----------------------------------------------------------------------------
// Port of the Pine Script IB Breakout Extension strategy.
//
// Concept
//   * IB = high/low of the first hour of RTH (configurable).
//   * After IB completes, two stop entries are armed (OCA-like behavior):
//        - Long  stop at IB high  (breakout)
//        - Short stop at IB low   (breakdown)
//     Whichever fills first becomes the trade; the other is canceled.
//   * If price later violates the OPPOSITE side of the IB, the day is
//     reclassified as a "double break" and the position is flattened.
//   * Scaled exits at IB-range extensions: 0.2 / 0.4 / 0.6 / 0.8 / 1.0 x IB.
//   * Optional breakeven stop triggered at a configurable tier.
//
// Session times use the CHART's timezone directly (no conversion).
//   - If your chart displays Eastern Time, use defaults (9:30 / 10:30 / 16:00).
//   - If your chart displays Central Time, set IB to 8:30-9:30 and RTH end 15:00.
//   - NinjaTrader bar timestamps (Time[0]) reflect the bar CLOSE time.
//     The time comparisons account for this automatically.
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class IBBreakoutExtensionStrategy : Strategy
    {
        public enum IBStopAnchorType
        {
            IBOpposite,
            IBMid,
            CustomIBMult
        }

        // ---- Runtime state ----
        private double ibHigh;
        private double ibLow;
        private double ibMid;
        private double ibRange;
        private bool   ibLocked;

        private bool brokeAbove;
        private bool brokeBelow;
        private bool doubleBreak;
        private bool beActive;

        private double longStopPx;
        private double shortStopPx;

        private double upL1, upL2, upL3, upL4, upL5;
        private double dnL1, dnL2, dnL3, dnL4, dnL5;

        private int    initialQty;
        private double entryPrice;

        private Order longEntryOrder;
        private Order shortEntryOrder;

        private int    ibFinishBarIndex;
        private string sessionTag;

        private Brush doubleBreakBrush;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                 = "IB Breakout Extension Strategy — scaled exits at IB-range extensions. "
                    + "Set IB/RTH hours to match your chart's timezone.";
                Name                        = "IBBreakoutExtensionStrategy";
                Calculate                   = Calculate.OnEachTick;
                EntriesPerDirection         = 1;
                EntryHandling               = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                ExitOnSessionCloseSeconds   = 30;
                IsFillLimitOnTouch          = false;
                MaximumBarsLookBack         = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution         = OrderFillResolution.Standard;
                Slippage                    = 0;
                StartBehavior               = StartBehavior.WaitUntilFlat;
                TimeInForce                 = TimeInForce.Gtc;
                TraceOrders                 = false;
                RealtimeErrorHandling       = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling          = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade         = 20;
                IsInstantiatedOnEachOptimizationIteration = true;

                IBStartHour         = 9;
                IBStartMinute       = 30;
                IBEndHour           = 10;
                IBEndMinute         = 30;
                RTHEndHour          = 16;
                RTHEndMinute        = 0;

                TradeLongs          = true;
                TradeShorts         = true;

                UseT1 = true;  Mult1 = 0.2;  Pct1 = 20;
                UseT2 = true;  Mult2 = 0.4;  Pct2 = 25;
                UseT3 = true;  Mult3 = 0.6;  Pct3 = 20;
                UseT4 = true;  Mult4 = 0.8;  Pct4 = 20;
                UseT5 = true;  Mult5 = 1.0;  Pct5 = 15;

                StopAnchor          = IBStopAnchorType.IBOpposite;
                CustomStopMultIB    = 0.5;
                FlattenOnDouble     = true;
                UseBreakeven        = true;
                BETier              = 2;
                ExitMinsBeforeClose = 15;

                MinIBRangePct       = 0.0;
                MaxIBRangePct       = 5.0;

                Contracts           = 10;

                ShowIB              = true;
                ShowTargets         = true;
                ShowStats           = true;
                SymProfile          = "NQ";
            }
            else if (State == State.DataLoaded)
            {
                ResetDayState();
                doubleBreakBrush = new SolidColorBrush(Color.FromArgb(25, 255, 165, 0));
                doubleBreakBrush.Freeze();
            }
        }

        private void ResetDayState()
        {
            ibHigh          = 0;
            ibLow           = 0;
            ibMid           = 0;
            ibRange         = 0;
            ibLocked        = false;
            brokeAbove      = false;
            brokeBelow      = false;
            doubleBreak     = false;
            beActive        = false;
            initialQty      = 0;
            entryPrice      = 0;
            longEntryOrder  = null;
            shortEntryOrder = null;
            ibFinishBarIndex = 0;
            sessionTag      = "";
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;
            if (CurrentBar < BarsRequiredToTrade) return;

            // ---- Session reset ----
            if (Bars.IsFirstBarOfSession)
                ResetDayState();

            // ---- Time detection (chart-native, no TZ conversion) ----
            int t    = ToTime(Time[0]);
            int ibSt = IBStartHour * 10000 + IBStartMinute * 100;
            int ibEn = IBEndHour   * 10000 + IBEndMinute   * 100;
            int rthEn = RTHEndHour * 10000 + RTHEndMinute  * 100;

            bool inIB  = t > ibSt  && t <= ibEn;
            bool inRTH = t > ibSt  && t <= rthEn;

            bool prevInIB  = false;
            bool prevInRTH = false;
            if (CurrentBar >= 1 && !Bars.IsFirstBarOfSession)
            {
                int tp = ToTime(Time[1]);
                prevInIB  = tp > ibSt  && tp <= ibEn;
                prevInRTH = tp > ibSt  && tp <= rthEn;
            }

            bool ibStart   = inIB  && !prevInIB;
            bool ibFinish  = !inIB && prevInIB && inRTH;
            bool rthStart  = inRTH && !prevInRTH;
            bool rthFinish = !inRTH && prevInRTH;

            // ---- IB tracking ----
            if (ibStart)
            {
                ibHigh   = High[0];
                ibLow    = Low[0];
                ibLocked = false;
            }
            else if (inIB)
            {
                ibHigh = Math.Max(ibHigh, High[0]);
                ibLow  = Math.Min(ibLow,  Low[0]);
            }

            if (ibFinish && ibHigh > 0 && ibLow > 0)
            {
                ibMid    = (ibHigh + ibLow) / 2.0;
                ibRange  = ibHigh - ibLow;
                ibLocked = true;
                ibFinishBarIndex = CurrentBar;
                sessionTag = Time[0].ToString("yyyyMMdd");

                // Compute targets once at IB lock
                upL1 = ibHigh + ibRange * Mult1;
                upL2 = ibHigh + ibRange * Mult2;
                upL3 = ibHigh + ibRange * Mult3;
                upL4 = ibHigh + ibRange * Mult4;
                upL5 = ibHigh + ibRange * Mult5;
                dnL1 = ibLow  - ibRange * Mult1;
                dnL2 = ibLow  - ibRange * Mult2;
                dnL3 = ibLow  - ibRange * Mult3;
                dnL4 = ibLow  - ibRange * Mult4;
                dnL5 = ibLow  - ibRange * Mult5;

                // Draw probability labels once at IB completion
                if (ShowStats)
                    DrawProbLabels();
            }

            if (rthStart)
            {
                brokeAbove  = false;
                brokeBelow  = false;
                doubleBreak = false;
                beActive    = false;
            }

            if (!ibLocked) return;

            // ---- Stop anchors ----
            switch (StopAnchor)
            {
                case IBStopAnchorType.IBOpposite:
                    longStopPx  = ibLow;
                    shortStopPx = ibHigh;
                    break;
                case IBStopAnchorType.IBMid:
                    longStopPx  = ibMid;
                    shortStopPx = ibMid;
                    break;
                default:
                    longStopPx  = ibHigh - ibRange * CustomStopMultIB;
                    shortStopPx = ibLow  + ibRange * CustomStopMultIB;
                    break;
            }

            // ---- Range filter ----
            double ibRangePct = (ibRange / Close[0]) * 100.0;
            bool rangeOK = ibRangePct >= MinIBRangePct && ibRangePct <= MaxIBRangePct;

            // ---- EOD guard ----
            int eodMins = (RTHEndHour * 60 + RTHEndMinute) - ExitMinsBeforeClose;
            int eodTime = (eodMins / 60) * 10000 + (eodMins % 60) * 100;
            bool flattenEOD = inRTH && t >= eodTime;

            bool canArm = ibLocked && inRTH && !doubleBreak && rangeOK && !flattenEOD
                          && Position.MarketPosition == MarketPosition.Flat;

            // ---- Entry arming ----
            if (canArm)
            {
                if (TradeLongs && !brokeAbove)
                    EnterLongStopMarket(0, true, Contracts, ibHigh, "L");
                if (TradeShorts && !brokeBelow)
                    EnterShortStopMarket(0, true, Contracts, ibLow, "S");
            }

            if (Position.MarketPosition == MarketPosition.Long && shortEntryOrder != null)
            {
                CancelOrder(shortEntryOrder);
                shortEntryOrder = null;
            }
            if (Position.MarketPosition == MarketPosition.Short && longEntryOrder != null)
            {
                CancelOrder(longEntryOrder);
                longEntryOrder = null;
            }

            // ---- Break tracking (AFTER entry arming) ----
            if (ibLocked && inRTH)
            {
                if (High[0] > ibHigh) brokeAbove = true;
                if (Low[0]  < ibLow)  brokeBelow = true;
                if (brokeAbove && brokeBelow) doubleBreak = true;
            }

            if (doubleBreak)
            {
                if (longEntryOrder != null)  { CancelOrder(longEntryOrder);  longEntryOrder  = null; }
                if (shortEntryOrder != null) { CancelOrder(shortEntryOrder); shortEntryOrder = null; }
            }

            // ---- Breakeven trigger detection ----
            if (UseBreakeven && !beActive && Position.MarketPosition != MarketPosition.Flat)
            {
                double bePxLong  = BETier == 1 ? upL1 : BETier == 2 ? upL2 : BETier == 3 ? upL3 : BETier == 4 ? upL4 : upL5;
                double bePxShort = BETier == 1 ? dnL1 : BETier == 2 ? dnL2 : BETier == 3 ? dnL3 : BETier == 4 ? dnL4 : dnL5;
                if (Position.MarketPosition == MarketPosition.Long  && High[0] >= bePxLong)  beActive = true;
                if (Position.MarketPosition == MarketPosition.Short && Low[0]  <= bePxShort) beActive = true;
            }

            double activeLongStop  = (UseBreakeven && beActive && entryPrice > 0) ? entryPrice : longStopPx;
            double activeShortStop = (UseBreakeven && beActive && entryPrice > 0) ? entryPrice : shortStopPx;

            // ---- Forced exits: double break + EOD ----
            if (FlattenOnDouble && doubleBreak && Position.MarketPosition != MarketPosition.Flat)
            {
                if (Position.MarketPosition == MarketPosition.Long)  ExitLong("DoubleBreakL",  "L");
                if (Position.MarketPosition == MarketPosition.Short) ExitShort("DoubleBreakS", "S");
                return;
            }

            if (flattenEOD && Position.MarketPosition != MarketPosition.Flat)
            {
                if (Position.MarketPosition == MarketPosition.Long)  ExitLong("EODL",  "L");
                if (Position.MarketPosition == MarketPosition.Short) ExitShort("EODS", "S");
                return;
            }

            if (rthFinish)
            {
                if (longEntryOrder != null)  { CancelOrder(longEntryOrder);  longEntryOrder  = null; }
                if (shortEntryOrder != null) { CancelOrder(shortEntryOrder); shortEntryOrder = null; }
            }

            // ---- Scaled exits ----
            if (Position.MarketPosition == MarketPosition.Long && initialQty > 0)
            {
                int q1 = TierQty(initialQty, Pct1);
                int q2 = TierQty(initialQty, Pct2);
                int q3 = TierQty(initialQty, Pct3);
                int q4 = TierQty(initialQty, Pct4);
                int q5 = TierQty(initialQty, Pct5);

                if (UseT1 && q1 > 0) ExitLongLimit(0, true, q1, upL1, "X1L", "L");
                if (UseT2 && q2 > 0) ExitLongLimit(0, true, q2, upL2, "X2L", "L");
                if (UseT3 && q3 > 0) ExitLongLimit(0, true, q3, upL3, "X3L", "L");
                if (UseT4 && q4 > 0) ExitLongLimit(0, true, q4, upL4, "X4L", "L");
                if (UseT5 && q5 > 0) ExitLongLimit(0, true, q5, upL5, "X5L", "L");

                int remaining = Math.Abs(Position.Quantity);
                if (remaining > 0)
                    ExitLongStopMarket(0, true, remaining, activeLongStop, "StopL", "L");
            }
            else if (Position.MarketPosition == MarketPosition.Short && initialQty > 0)
            {
                int q1 = TierQty(initialQty, Pct1);
                int q2 = TierQty(initialQty, Pct2);
                int q3 = TierQty(initialQty, Pct3);
                int q4 = TierQty(initialQty, Pct4);
                int q5 = TierQty(initialQty, Pct5);

                if (UseT1 && q1 > 0) ExitShortLimit(0, true, q1, dnL1, "X1S", "S");
                if (UseT2 && q2 > 0) ExitShortLimit(0, true, q2, dnL2, "X2S", "S");
                if (UseT3 && q3 > 0) ExitShortLimit(0, true, q3, dnL3, "X3S", "S");
                if (UseT4 && q4 > 0) ExitShortLimit(0, true, q4, dnL4, "X4S", "S");
                if (UseT5 && q5 > 0) ExitShortLimit(0, true, q5, dnL5, "X5S", "S");

                int remaining = Math.Abs(Position.Quantity);
                if (remaining > 0)
                    ExitShortStopMarket(0, true, remaining, activeShortStop, "StopS", "S");
            }

            // ---- Visualization (once per bar for performance) ----
            if (IsFirstTickOfBar && ibLocked && sessionTag.Length > 0)
            {
                int bb = CurrentBar - ibFinishBarIndex;
                if (bb < 0) bb = 0;

                if (ShowIB)
                {
                    Draw.Line(this, "IBH" + sessionTag, false, bb, ibHigh, 0, ibHigh,
                        Brushes.Aqua, DashStyleHelper.Solid, 2);
                    Draw.Line(this, "IBL" + sessionTag, false, bb, ibLow, 0, ibLow,
                        Brushes.Fuchsia, DashStyleHelper.Solid, 2);
                    Draw.Line(this, "IBM" + sessionTag, false, bb, ibMid, 0, ibMid,
                        Brushes.Gray, DashStyleHelper.Dash, 1);
                }

                if (ShowTargets)
                {
                    if (UseT1)
                    {
                        Draw.Line(this, "U1" + sessionTag, false, bb, upL1, 0, upL1,
                            Brushes.Lime, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D1" + sessionTag, false, bb, dnL1, 0, dnL1,
                            Brushes.Red, DashStyleHelper.Dot, 1);
                    }
                    if (UseT2)
                    {
                        Draw.Line(this, "U2" + sessionTag, false, bb, upL2, 0, upL2,
                            Brushes.Lime, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D2" + sessionTag, false, bb, dnL2, 0, dnL2,
                            Brushes.Red, DashStyleHelper.Dot, 1);
                    }
                    if (UseT3)
                    {
                        Draw.Line(this, "U3" + sessionTag, false, bb, upL3, 0, upL3,
                            Brushes.LimeGreen, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D3" + sessionTag, false, bb, dnL3, 0, dnL3,
                            Brushes.OrangeRed, DashStyleHelper.Dot, 1);
                    }
                    if (UseT4)
                    {
                        Draw.Line(this, "U4" + sessionTag, false, bb, upL4, 0, upL4,
                            Brushes.LimeGreen, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D4" + sessionTag, false, bb, dnL4, 0, dnL4,
                            Brushes.OrangeRed, DashStyleHelper.Dot, 1);
                    }
                    if (UseT5)
                    {
                        Draw.Line(this, "U5" + sessionTag, false, bb, upL5, 0, upL5,
                            Brushes.DarkGreen, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D5" + sessionTag, false, bb, dnL5, 0, dnL5,
                            Brushes.DarkRed, DashStyleHelper.Dot, 1);
                    }
                }

                if (doubleBreak)
                    BackBrushes[0] = doubleBreakBrush;
            }
        }

        // ---- Helpers ----
        private static int TierQty(int total, double pct)
        {
            int q = (int)Math.Floor(total * pct / 100.0);
            return q < 0 ? 0 : q;
        }

        private double GetProb(int idx)
        {
            switch (SymProfile)
            {
                case "ES":
                    return idx == 0 ? 92.2 : idx == 1 ? 77.7 : idx == 2 ? 60.6 : idx == 3 ? 52.4 : 39.4;
                case "YM":
                    return idx == 0 ? 86.3 : idx == 1 ? 69.1 : idx == 2 ? 51.2 : idx == 3 ? 40.2 : 29.2;
                default:
                    return idx == 0 ? 87.5 : idx == 1 ? 70.3 : idx == 2 ? 52.1 : idx == 3 ? 36.1 : 25.2;
            }
        }

        private void DrawProbLabels()
        {
            string s = sessionTag;
            if (UseT1)
            {
                Draw.Text(this, "PU1" + s, "+0.2  " + GetProb(0).ToString("0.#") + "%", 0, upL1, Brushes.Lime);
                Draw.Text(this, "PD1" + s, "-0.2  " + GetProb(0).ToString("0.#") + "%", 0, dnL1, Brushes.Red);
            }
            if (UseT2)
            {
                Draw.Text(this, "PU2" + s, "+0.4  " + GetProb(1).ToString("0.#") + "%", 0, upL2, Brushes.Lime);
                Draw.Text(this, "PD2" + s, "-0.4  " + GetProb(1).ToString("0.#") + "%", 0, dnL2, Brushes.Red);
            }
            if (UseT3)
            {
                Draw.Text(this, "PU3" + s, "+0.6  " + GetProb(2).ToString("0.#") + "%", 0, upL3, Brushes.LimeGreen);
                Draw.Text(this, "PD3" + s, "-0.6  " + GetProb(2).ToString("0.#") + "%", 0, dnL3, Brushes.OrangeRed);
            }
            if (UseT4)
            {
                Draw.Text(this, "PU4" + s, "+0.8  " + GetProb(3).ToString("0.#") + "%", 0, upL4, Brushes.LimeGreen);
                Draw.Text(this, "PD4" + s, "-0.8  " + GetProb(3).ToString("0.#") + "%", 0, dnL4, Brushes.OrangeRed);
            }
            if (UseT5)
            {
                Draw.Text(this, "PU5" + s, "+1.0  " + GetProb(4).ToString("0.#") + "%", 0, upL5, Brushes.DarkGreen);
                Draw.Text(this, "PD5" + s, "-1.0  " + GetProb(4).ToString("0.#") + "%", 0, dnL5, Brushes.DarkRed);
            }
        }

        // ---- Order / Execution callbacks ----
        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity,
            int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            if (order.Name == "L" && order.OrderState == OrderState.Working)
                longEntryOrder = order;
            if (order.Name == "S" && order.OrderState == OrderState.Working)
                shortEntryOrder = order;

            if (order.Name == "L" && (order.OrderState == OrderState.Cancelled || order.OrderState == OrderState.Filled
                || order.OrderState == OrderState.Rejected))
                if (order == longEntryOrder) longEntryOrder = null;
            if (order.Name == "S" && (order.OrderState == OrderState.Cancelled || order.OrderState == OrderState.Filled
                || order.OrderState == OrderState.Rejected))
                if (order == shortEntryOrder) shortEntryOrder = null;
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution.Order == null) return;
            if (execution.Order.OrderState != OrderState.Filled && execution.Order.OrderState != OrderState.PartFilled)
                return;

            if (execution.Order.Name == "L" || execution.Order.Name == "S")
            {
                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    initialQty = Math.Abs(Position.Quantity);
                    entryPrice = Position.AveragePrice;
                }
            }

            if (Position.MarketPosition == MarketPosition.Flat)
            {
                initialQty = 0;
                entryPrice = 0;
                beActive   = false;
            }
        }

        // =====================================================================
        // Properties
        // =====================================================================

        // -- Session --
        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "IB Start Hour",   GroupName = "1. Session", Order = 1,
            Description = "Hour the IB window opens, in your chart's timezone. ET=9, CT=8.")]
        public int IBStartHour { get; set; }

        [NinjaScriptProperty]
        [Range(0, 59)]
        [Display(Name = "IB Start Minute", GroupName = "1. Session", Order = 2)]
        public int IBStartMinute { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "IB End Hour",     GroupName = "1. Session", Order = 3,
            Description = "Hour the IB window closes, in your chart's timezone. ET=10, CT=9.")]
        public int IBEndHour { get; set; }

        [NinjaScriptProperty]
        [Range(0, 59)]
        [Display(Name = "IB End Minute",   GroupName = "1. Session", Order = 4)]
        public int IBEndMinute { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "RTH End Hour",    GroupName = "1. Session", Order = 5,
            Description = "Hour RTH closes, in your chart's timezone. ET=16, CT=15.")]
        public int RTHEndHour { get; set; }

        [NinjaScriptProperty]
        [Range(0, 59)]
        [Display(Name = "RTH End Minute",  GroupName = "1. Session", Order = 6)]
        public int RTHEndMinute { get; set; }

        // -- Direction --
        [NinjaScriptProperty]
        [Display(Name = "Take Breakout Longs",   GroupName = "2. Trade Direction", Order = 1)]
        public bool TradeLongs { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Take Breakdown Shorts", GroupName = "2. Trade Direction", Order = 2)]
        public bool TradeShorts { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Contracts per Trade", GroupName = "2. Trade Direction", Order = 3,
            Description = "Use >= 7 for all 5 tiers to get at least 1 contract per tier.")]
        public int Contracts { get; set; }

        // -- Targets --
        [NinjaScriptProperty] [Display(Name = "Use T1", GroupName = "3. Targets", Order = 1)]  public bool   UseT1 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T1 Mult",      GroupName = "3. Targets", Order = 2)]  public double Mult1 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T1 % Out",     GroupName = "3. Targets", Order = 3)]  public double Pct1  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T2", GroupName = "3. Targets", Order = 4)]  public bool   UseT2 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T2 Mult",      GroupName = "3. Targets", Order = 5)]  public double Mult2 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T2 % Out",     GroupName = "3. Targets", Order = 6)]  public double Pct2  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T3", GroupName = "3. Targets", Order = 7)]  public bool   UseT3 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T3 Mult",      GroupName = "3. Targets", Order = 8)]  public double Mult3 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T3 % Out",     GroupName = "3. Targets", Order = 9)]  public double Pct3  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T4", GroupName = "3. Targets", Order = 10)] public bool   UseT4 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T4 Mult",      GroupName = "3. Targets", Order = 11)] public double Mult4 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T4 % Out",     GroupName = "3. Targets", Order = 12)] public double Pct4  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T5", GroupName = "3. Targets", Order = 13)] public bool   UseT5 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T5 Mult",      GroupName = "3. Targets", Order = 14)] public double Mult5 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T5 % Out",     GroupName = "3. Targets", Order = 15)] public double Pct5  { get; set; }

        // -- Risk --
        [NinjaScriptProperty]
        [Display(Name = "Stop Anchor", GroupName = "4. Risk", Order = 1)]
        public IBStopAnchorType StopAnchor { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Custom Stop (x IB Range)", GroupName = "4. Risk", Order = 2)]
        public double CustomStopMultIB { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Flatten on Double Break", GroupName = "4. Risk", Order = 3)]
        public bool FlattenOnDouble { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Move Stop to Breakeven", GroupName = "4. Risk", Order = 4)]
        public bool UseBreakeven { get; set; }

        [NinjaScriptProperty]
        [Range(1, 5)]
        [Display(Name = "BE Trigger Tier (1-5)", GroupName = "4. Risk", Order = 5)]
        public int BETier { get; set; }

        [NinjaScriptProperty]
        [Range(0, 240)]
        [Display(Name = "Flatten N Minutes Before Close", GroupName = "4. Risk", Order = 6)]
        public int ExitMinsBeforeClose { get; set; }

        // -- Filters --
        [NinjaScriptProperty]
        [Display(Name = "Min IB Range (% of price)", GroupName = "5. Filters", Order = 1)]
        public double MinIBRangePct { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max IB Range (% of price)", GroupName = "5. Filters", Order = 2)]
        public double MaxIBRangePct { get; set; }

        // -- Visualization --
        [NinjaScriptProperty]
        [Display(Name = "Draw IB High/Low/Mid", GroupName = "6. Visualization", Order = 1)]
        public bool ShowIB { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Draw Target Levels", GroupName = "6. Visualization", Order = 2)]
        public bool ShowTargets { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Probability Labels", GroupName = "6. Visualization", Order = 3)]
        public bool ShowStats { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Probability Profile (NQ/ES/YM)", GroupName = "6. Visualization", Order = 4,
            Description = "Which instrument's hit-rate stats to display on labels.")]
        public string SymProfile { get; set; }
    }
}
