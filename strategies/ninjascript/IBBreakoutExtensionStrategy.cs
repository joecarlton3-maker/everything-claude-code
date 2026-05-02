// =============================================================================
// IB Breakout Extension Strategy  (NQ / ES / YM)  -  NinjaTrader 8 / NinjaScript
// -----------------------------------------------------------------------------
// Port of the Pine Script IB Breakout Extension strategy.
//
// Concept
//   * IB = high/low of the first hour of RTH (configurable).
//   * After IB completes, stop entries are armed per tier (OCA-like behavior):
//        - Long  stops at IB high  (breakout)  — one per active tier
//        - Short stops at IB low   (breakdown)  — one per active tier
//     Whichever SIDE fills first becomes the trade; the other side is canceled.
//   * Each tier entry gets its own paired bracket (limit target + stop loss).
//     This ensures contracts scale out correctly across all 5 extension levels.
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

        // Pre-computed signal names to avoid per-tick string allocation
        private static readonly string[] LN  = { "L1", "L2", "L3", "L4", "L5" };
        private static readonly string[] SN  = { "S1", "S2", "S3", "S4", "S5" };
        private static readonly string[] XLN = { "XL1", "XL2", "XL3", "XL4", "XL5" };
        private static readonly string[] SLN = { "SL1", "SL2", "SL3", "SL4", "SL5" };
        private static readonly string[] XSN = { "XS1", "XS2", "XS3", "XS4", "XS5" };
        private static readonly string[] SSN = { "SS1", "SS2", "SS3", "SS4", "SS5" };

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

        private double[] upTgt;
        private double[] dnTgt;
        private int[]    tierQty;

        private double entryPrice;

        private Order[] longEntryOrders;
        private Order[] shortEntryOrders;

        private int    ibFinishBarIndex;
        private string sessionTag;

        private Brush doubleBreakBrush;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                 = "IB Breakout Extension Strategy — scaled exits at IB-range extensions. "
                    + "Uses per-tier sub-entries so each tier bracket (target + stop) is independent.";
                Name                        = "IBBreakoutExtensionStrategy";
                Calculate                   = Calculate.OnEachTick;
                EntriesPerDirection         = 5;
                EntryHandling               = EntryHandling.UniqueEntries;
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
                upTgt            = new double[5];
                dnTgt            = new double[5];
                tierQty          = new int[5];
                longEntryOrders  = new Order[5];
                shortEntryOrders = new Order[5];
                ResetDayState();
                doubleBreakBrush = new SolidColorBrush(Color.FromArgb(25, 255, 165, 0));
                doubleBreakBrush.Freeze();
            }
        }

        private void ResetDayState()
        {
            ibHigh          = double.MinValue;
            ibLow           = double.MaxValue;
            ibMid           = 0;
            ibRange         = 0;
            ibLocked        = false;
            brokeAbove      = false;
            brokeBelow      = false;
            doubleBreak     = false;
            beActive        = false;
            entryPrice      = 0;
            ibFinishBarIndex = 0;
            sessionTag      = "";
            for (int i = 0; i < 5; i++)
            {
                upTgt[i]            = 0;
                dnTgt[i]            = 0;
                tierQty[i]          = 0;
                longEntryOrders[i]  = null;
                shortEntryOrders[i] = null;
            }
        }

        private void ComputeTierQty()
        {
            bool[]   use = { UseT1, UseT2, UseT3, UseT4, UseT5 };
            double[] pct = { Pct1,  Pct2,  Pct3,  Pct4,  Pct5  };

            int total = 0;
            for (int i = 0; i < 5; i++)
            {
                tierQty[i] = use[i] ? (int)Math.Floor(Contracts * pct[i] / 100.0) : 0;
                total += tierQty[i];
            }

            int rem = Contracts - total;
            for (int i = 4; i >= 0 && rem > 0; i--)
            {
                if (use[i])
                {
                    tierQty[i] += rem;
                    rem = 0;
                }
            }
        }

        private void CancelAllEntries()
        {
            for (int i = 0; i < 5; i++)
            {
                if (longEntryOrders[i] != null)
                {
                    CancelOrder(longEntryOrders[i]);
                    longEntryOrders[i] = null;
                }
                if (shortEntryOrders[i] != null)
                {
                    CancelOrder(shortEntryOrders[i]);
                    shortEntryOrders[i] = null;
                }
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;
            if (CurrentBar < BarsRequiredToTrade) return;

            // ---- Session reset ----
            if (Bars.IsFirstBarOfSession)
                ResetDayState();

            // ---- Time detection (chart-native, no TZ conversion) ----
            int t     = ToTime(Time[0]);
            int ibSt  = IBStartHour * 10000 + IBStartMinute * 100;
            int ibEn  = IBEndHour   * 10000 + IBEndMinute   * 100;
            int rthEn = RTHEndHour  * 10000 + RTHEndMinute  * 100;

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

            if (ibFinish && ibHigh > ibLow)
            {
                ibMid    = (ibHigh + ibLow) / 2.0;
                ibRange  = ibHigh - ibLow;
                ibLocked = true;
                ibFinishBarIndex = CurrentBar;
                sessionTag = Time[0].ToString("yyyyMMdd");

                double[] mult = { Mult1, Mult2, Mult3, Mult4, Mult5 };
                for (int i = 0; i < 5; i++)
                {
                    upTgt[i] = ibHigh + ibRange * mult[i];
                    dnTgt[i] = ibLow  - ibRange * mult[i];
                }

                ComputeTierQty();

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

            // ---- Entry arming: one sub-entry per active tier ----
            if (canArm)
            {
                if (TradeLongs && !brokeAbove)
                {
                    for (int i = 0; i < 5; i++)
                        if (tierQty[i] > 0)
                            EnterLongStopMarket(0, true, tierQty[i], ibHigh, LN[i]);
                }
                if (TradeShorts && !brokeBelow)
                {
                    for (int i = 0; i < 5; i++)
                        if (tierQty[i] > 0)
                            EnterShortStopMarket(0, true, tierQty[i], ibLow, SN[i]);
                }
            }

            // ---- OCA emulation: cancel opposite side on fill ----
            if (Position.MarketPosition == MarketPosition.Long)
            {
                for (int i = 0; i < 5; i++)
                {
                    if (shortEntryOrders[i] != null)
                    {
                        CancelOrder(shortEntryOrders[i]);
                        shortEntryOrders[i] = null;
                    }
                }
            }
            if (Position.MarketPosition == MarketPosition.Short)
            {
                for (int i = 0; i < 5; i++)
                {
                    if (longEntryOrders[i] != null)
                    {
                        CancelOrder(longEntryOrders[i]);
                        longEntryOrders[i] = null;
                    }
                }
            }

            // ---- Break tracking (AFTER entry arming) ----
            if (ibLocked && inRTH)
            {
                if (High[0] > ibHigh) brokeAbove = true;
                if (Low[0]  < ibLow)  brokeBelow = true;
                if (brokeAbove && brokeBelow) doubleBreak = true;
            }

            if (doubleBreak)
                CancelAllEntries();

            // ---- Breakeven trigger detection ----
            if (UseBreakeven && !beActive && Position.MarketPosition != MarketPosition.Flat)
            {
                int bi = BETier - 1;
                if (bi >= 0 && bi < 5)
                {
                    if (Position.MarketPosition == MarketPosition.Long  && High[0] >= upTgt[bi])
                        beActive = true;
                    if (Position.MarketPosition == MarketPosition.Short && Low[0]  <= dnTgt[bi])
                        beActive = true;
                }
            }

            double activeLongStop  = (UseBreakeven && beActive && entryPrice > 0) ? entryPrice : longStopPx;
            double activeShortStop = (UseBreakeven && beActive && entryPrice > 0) ? entryPrice : shortStopPx;

            // ---- Forced exits: double break ----
            if (FlattenOnDouble && doubleBreak && Position.MarketPosition != MarketPosition.Flat)
            {
                for (int i = 0; i < 5; i++)
                {
                    if (tierQty[i] > 0)
                    {
                        if (Position.MarketPosition == MarketPosition.Long)
                            ExitLong("DBL" + (i + 1), LN[i]);
                        else
                            ExitShort("DBS" + (i + 1), SN[i]);
                    }
                }
                return;
            }

            // ---- Forced exits: EOD ----
            if (flattenEOD)
            {
                CancelAllEntries();
                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    for (int i = 0; i < 5; i++)
                    {
                        if (tierQty[i] > 0)
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong("EODL" + (i + 1), LN[i]);
                            else
                                ExitShort("EODS" + (i + 1), SN[i]);
                        }
                    }
                    return;
                }
            }

            // ---- RTH finish: cancel entries + safety flatten ----
            if (rthFinish)
            {
                CancelAllEntries();
                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    for (int i = 0; i < 5; i++)
                    {
                        if (tierQty[i] > 0)
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong("RTHL" + (i + 1), LN[i]);
                            else
                                ExitShort("RTHS" + (i + 1), SN[i]);
                        }
                    }
                }
                return;
            }

            // ---- Scaled exits: paired limit + stop per sub-entry ----
            if (Position.MarketPosition == MarketPosition.Long)
            {
                for (int i = 0; i < 5; i++)
                {
                    if (tierQty[i] > 0)
                    {
                        ExitLongLimit(0, true, tierQty[i], upTgt[i], XLN[i], LN[i]);
                        ExitLongStopMarket(0, true, tierQty[i], activeLongStop, SLN[i], LN[i]);
                    }
                }
            }
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                for (int i = 0; i < 5; i++)
                {
                    if (tierQty[i] > 0)
                    {
                        ExitShortLimit(0, true, tierQty[i], dnTgt[i], XSN[i], SN[i]);
                        ExitShortStopMarket(0, true, tierQty[i], activeShortStop, SSN[i], SN[i]);
                    }
                }
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
                        Draw.Line(this, "U1" + sessionTag, false, bb, upTgt[0], 0, upTgt[0],
                            Brushes.Lime, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D1" + sessionTag, false, bb, dnTgt[0], 0, dnTgt[0],
                            Brushes.Red, DashStyleHelper.Dot, 1);
                    }
                    if (UseT2)
                    {
                        Draw.Line(this, "U2" + sessionTag, false, bb, upTgt[1], 0, upTgt[1],
                            Brushes.Lime, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D2" + sessionTag, false, bb, dnTgt[1], 0, dnTgt[1],
                            Brushes.Red, DashStyleHelper.Dot, 1);
                    }
                    if (UseT3)
                    {
                        Draw.Line(this, "U3" + sessionTag, false, bb, upTgt[2], 0, upTgt[2],
                            Brushes.LimeGreen, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D3" + sessionTag, false, bb, dnTgt[2], 0, dnTgt[2],
                            Brushes.OrangeRed, DashStyleHelper.Dot, 1);
                    }
                    if (UseT4)
                    {
                        Draw.Line(this, "U4" + sessionTag, false, bb, upTgt[3], 0, upTgt[3],
                            Brushes.LimeGreen, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D4" + sessionTag, false, bb, dnTgt[3], 0, dnTgt[3],
                            Brushes.OrangeRed, DashStyleHelper.Dot, 1);
                    }
                    if (UseT5)
                    {
                        Draw.Line(this, "U5" + sessionTag, false, bb, upTgt[4], 0, upTgt[4],
                            Brushes.DarkGreen, DashStyleHelper.Dot, 1);
                        Draw.Line(this, "D5" + sessionTag, false, bb, dnTgt[4], 0, dnTgt[4],
                            Brushes.DarkRed, DashStyleHelper.Dot, 1);
                    }
                }

                if (doubleBreak)
                    BackBrushes[0] = doubleBreakBrush;
            }
        }

        // ---- Helpers ----

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
            string[] upStr = { "+0.2", "+0.4", "+0.6", "+0.8", "+1.0" };
            string[] dnStr = { "-0.2", "-0.4", "-0.6", "-0.8", "-1.0" };
            Brush[]  upBr  = { Brushes.Lime, Brushes.Lime, Brushes.LimeGreen, Brushes.LimeGreen, Brushes.DarkGreen };
            Brush[]  dnBr  = { Brushes.Red,  Brushes.Red,  Brushes.OrangeRed, Brushes.OrangeRed, Brushes.DarkRed   };
            bool[]   use   = { UseT1, UseT2, UseT3, UseT4, UseT5 };

            for (int i = 0; i < 5; i++)
            {
                if (use[i])
                {
                    string prob = GetProb(i).ToString("0.#") + "%";
                    Draw.Text(this, "PU" + (i + 1) + s, upStr[i] + "  " + prob, 0, upTgt[i], upBr[i]);
                    Draw.Text(this, "PD" + (i + 1) + s, dnStr[i] + "  " + prob, 0, dnTgt[i], dnBr[i]);
                }
            }
        }

        // ---- Order / Execution callbacks ----
        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity,
            int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            for (int i = 0; i < 5; i++)
            {
                if (order.Name == LN[i])
                {
                    if (order.OrderState == OrderState.Working || order.OrderState == OrderState.Accepted)
                        longEntryOrders[i] = order;
                    if (order.OrderState == OrderState.Cancelled || order.OrderState == OrderState.Filled
                        || order.OrderState == OrderState.Rejected)
                        longEntryOrders[i] = null;
                }
                if (order.Name == SN[i])
                {
                    if (order.OrderState == OrderState.Working || order.OrderState == OrderState.Accepted)
                        shortEntryOrders[i] = order;
                    if (order.OrderState == OrderState.Cancelled || order.OrderState == OrderState.Filled
                        || order.OrderState == OrderState.Rejected)
                        shortEntryOrders[i] = null;
                }
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution.Order == null) return;
            if (execution.Order.OrderState != OrderState.Filled && execution.Order.OrderState != OrderState.PartFilled)
                return;

            string name = execution.Order.Name;
            bool isEntry = false;
            for (int i = 0; i < 5; i++)
            {
                if (name == LN[i] || name == SN[i])
                {
                    isEntry = true;
                    break;
                }
            }

            if (isEntry && Position.MarketPosition != MarketPosition.Flat)
                entryPrice = Position.AveragePrice;

            if (Position.MarketPosition == MarketPosition.Flat)
            {
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
