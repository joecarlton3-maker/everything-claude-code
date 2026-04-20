// =============================================================================
// IB Breakout Extension Strategy  (NQ / ES / YM)  -  NinjaTrader 8 / NinjaScript
// -----------------------------------------------------------------------------
// Port of the Pine Script IB Breakout Extension strategy.
//
// Concept
//   * IB = high/low of 09:30-10:30 ET (configurable).
//   * After IB completes, two stop entries are armed (OCA-like behavior):
//        - Long  stop at IB high  (breakout)
//        - Short stop at IB low   (breakdown)
//     Whichever fills first becomes the trade; the other is canceled.
//   * If price later violates the OPPOSITE side of the IB, the day is
//     reclassified as a "double break" and the position is flattened.
//   * Scaled exits at IB-range extensions: 0.2 / 0.4 / 0.6 / 0.8 / 1.0 x IB.
//   * Optional breakeven stop: once the configured tier fills, move the
//     stop for all remaining exits to the entry price.
//
// NinjaTrader Notes
//   * NinjaScript does not have native OCA, so we cancel the opposite
//     pending entry inside OnOrderUpdate/OnBarUpdate when the other fills.
//   * Scaled exits are submitted as separate ExitLongLimit/ExitShortLimit
//     signals with fixed per-tier quantities computed from the position's
//     initial size.
//   * The protective stop is re-submitted each bar via ExitLongStopMarket
//     / ExitShortStopMarket on the remaining position size.  When BE
//     triggers, the stop price is swapped to the average entry price.
//   * Designed for Calculate.OnEachTick; also works on OnBarClose with
//     1-minute bars.  Prefer OnEachTick for accurate intrabar tier fills.
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public enum IBStopAnchorType
    {
        IBOpposite,
        IBMid,
        CustomIBMult
    }

    public class IBBreakoutExtensionStrategy : Strategy
    {
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

        private DateTime lastSessionDate = DateTime.MinValue;
        private TimeZoneInfo easternTz;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                 = "IB Breakout Extension Strategy — scaled exits at IB-range extensions.";
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

                // ---- User inputs (defaults match the Pine Script) ----
                IBStartHour        = 9;
                IBStartMinute      = 30;
                IBEndHour          = 10;
                IBEndMinute        = 30;
                RTHEndHour         = 16;
                RTHEndMinute       = 0;
                SessionTimeZoneId  = "Eastern Standard Time";

                TradeLongs         = true;
                TradeShorts        = true;

                UseT1 = true; Mult1 = 0.2; Pct1 = 20;
                UseT2 = true; Mult2 = 0.4; Pct2 = 25;
                UseT3 = true; Mult3 = 0.6; Pct3 = 20;
                UseT4 = true; Mult4 = 0.8; Pct4 = 20;
                UseT5 = true; Mult5 = 1.0; Pct5 = 15;

                StopAnchor         = IBStopAnchorType.IBOpposite;
                CustomStopMultIB   = 0.5;
                FlattenOnDouble    = true;
                UseBreakeven       = true;
                BETier             = 2;
                ExitMinsBeforeClose = 15;

                MinIBRangePct      = 0.0;
                MaxIBRangePct      = 5.0;

                Contracts          = 10;
            }
            else if (State == State.Configure)
            {
                try
                {
                    easternTz = TimeZoneInfo.FindSystemTimeZoneById(SessionTimeZoneId);
                }
                catch
                {
                    easternTz = TimeZoneInfo.Local;
                }
            }
            else if (State == State.DataLoaded)
            {
                ResetDayState();
            }
        }

        private void ResetDayState()
        {
            ibHigh = 0;
            ibLow = 0;
            ibMid = 0;
            ibRange = 0;
            ibLocked = false;
            brokeAbove = false;
            brokeBelow = false;
            doubleBreak = false;
            beActive = false;
            initialQty = 0;
            entryPrice = 0;
            longEntryOrder = null;
            shortEntryOrder = null;
        }

        private DateTime ToEastern(DateTime t)
        {
            try
            {
                DateTime utc = t.Kind == DateTimeKind.Utc ? t : TimeZoneInfo.ConvertTimeToUtc(t, TimeZoneInfo.Local);
                return TimeZoneInfo.ConvertTimeFromUtc(utc, easternTz);
            }
            catch
            {
                return t;
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;
            if (CurrentBar < BarsRequiredToTrade) return;

            DateTime et = ToEastern(Time[0]);

            if (et.Date != lastSessionDate)
            {
                ResetDayState();
                lastSessionDate = et.Date;
            }

            int barMins      = et.Hour * 60 + et.Minute;
            int ibStartMins  = IBStartHour * 60 + IBStartMinute;
            int ibEndMins    = IBEndHour   * 60 + IBEndMinute;
            int rthStartMins = ibStartMins;
            int rthEndMins   = RTHEndHour  * 60 + RTHEndMinute;

            bool inIB  = barMins >= ibStartMins  && barMins <  ibEndMins;
            bool inRTH = barMins >= rthStartMins && barMins <  rthEndMins;

            bool prevInIB  = false;
            bool prevInRTH = false;
            if (CurrentBar >= 1)
            {
                DateTime etPrev = ToEastern(Time[1]);
                int prevMins = etPrev.Hour * 60 + etPrev.Minute;
                bool sameDay = etPrev.Date == et.Date;
                prevInIB  = sameDay && prevMins >= ibStartMins  && prevMins <  ibEndMins;
                prevInRTH = sameDay && prevMins >= rthStartMins && prevMins <  rthEndMins;
            }
            bool ibStart   = inIB  && !prevInIB;
            bool ibFinish  = !inIB && prevInIB;
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

            if (ibFinish)
            {
                ibMid    = (ibHigh + ibLow) / 2.0;
                ibRange  = ibHigh - ibLow;
                ibLocked = true;
            }

            if (rthStart)
            {
                brokeAbove  = false;
                brokeBelow  = false;
                doubleBreak = false;
                beActive    = false;
            }

            if (!ibLocked) return;

            // ---- Extension targets ----
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
            int closeMin = rthEndMins - ExitMinsBeforeClose;
            bool flattenEOD = inRTH && barMins >= closeMin;

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

            // Cancel opposite pending entry once we're in a trade
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
        }

        private static int TierQty(int total, double pct)
        {
            int q = (int)Math.Floor(total * pct / 100.0);
            return q < 0 ? 0 : q;
        }

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

            // Capture initial position size + entry price on entry fill
            if (execution.Order.Name == "L" || execution.Order.Name == "S")
            {
                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    initialQty = Math.Abs(Position.Quantity);
                    entryPrice = Position.AveragePrice;
                }
            }

            // Reset on full exit
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                initialQty = 0;
                entryPrice = 0;
                beActive = false;
            }
        }

        // =====================================================================
        // Properties
        // =====================================================================
        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "IB Start Hour (ET)",   GroupName = "Session", Order = 1)]
        public int IBStartHour { get; set; }

        [NinjaScriptProperty]
        [Range(0, 59)]
        [Display(Name = "IB Start Minute (ET)", GroupName = "Session", Order = 2)]
        public int IBStartMinute { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "IB End Hour (ET)",     GroupName = "Session", Order = 3)]
        public int IBEndHour { get; set; }

        [NinjaScriptProperty]
        [Range(0, 59)]
        [Display(Name = "IB End Minute (ET)",   GroupName = "Session", Order = 4)]
        public int IBEndMinute { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "RTH End Hour (ET)",    GroupName = "Session", Order = 5)]
        public int RTHEndHour { get; set; }

        [NinjaScriptProperty]
        [Range(0, 59)]
        [Display(Name = "RTH End Minute (ET)",  GroupName = "Session", Order = 6)]
        public int RTHEndMinute { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Session Time Zone Id", GroupName = "Session", Order = 7,
            Description = "Windows time zone id. Default: Eastern Standard Time")]
        public string SessionTimeZoneId { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Take Breakout Longs",   GroupName = "Trade Direction", Order = 1)]
        public bool TradeLongs { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Take Breakdown Shorts", GroupName = "Trade Direction", Order = 2)]
        public bool TradeShorts { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Contracts per Trade", GroupName = "Trade Direction", Order = 3)]
        public int Contracts { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T1 (0.2)", GroupName = "Targets", Order = 1)]  public bool   UseT1 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T1 Mult",      GroupName = "Targets", Order = 2)]  public double Mult1 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T1 % Out",     GroupName = "Targets", Order = 3)]  public double Pct1  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T2 (0.4)", GroupName = "Targets", Order = 4)]  public bool   UseT2 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T2 Mult",      GroupName = "Targets", Order = 5)]  public double Mult2 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T2 % Out",     GroupName = "Targets", Order = 6)]  public double Pct2  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T3 (0.6)", GroupName = "Targets", Order = 7)]  public bool   UseT3 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T3 Mult",      GroupName = "Targets", Order = 8)]  public double Mult3 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T3 % Out",     GroupName = "Targets", Order = 9)]  public double Pct3  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T4 (0.8)", GroupName = "Targets", Order = 10)] public bool   UseT4 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T4 Mult",      GroupName = "Targets", Order = 11)] public double Mult4 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T4 % Out",     GroupName = "Targets", Order = 12)] public double Pct4  { get; set; }

        [NinjaScriptProperty] [Display(Name = "Use T5 (1.0)", GroupName = "Targets", Order = 13)] public bool   UseT5 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T5 Mult",      GroupName = "Targets", Order = 14)] public double Mult5 { get; set; }
        [NinjaScriptProperty] [Display(Name = "T5 % Out",     GroupName = "Targets", Order = 15)] public double Pct5  { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Stop Anchor", GroupName = "Risk", Order = 1)]
        public IBStopAnchorType StopAnchor { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Custom Stop (x IB Range)", GroupName = "Risk", Order = 2)]
        public double CustomStopMultIB { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Flatten on Double Break", GroupName = "Risk", Order = 3)]
        public bool FlattenOnDouble { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Move Stop to Breakeven", GroupName = "Risk", Order = 4)]
        public bool UseBreakeven { get; set; }

        [NinjaScriptProperty]
        [Range(1, 5)]
        [Display(Name = "BE Trigger Tier (1-5)", GroupName = "Risk", Order = 5)]
        public int BETier { get; set; }

        [NinjaScriptProperty]
        [Range(0, 240)]
        [Display(Name = "Flatten N Minutes Before Close", GroupName = "Risk", Order = 6)]
        public int ExitMinsBeforeClose { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Min IB Range (% of price)", GroupName = "Filters", Order = 1)]
        public double MinIBRangePct { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max IB Range (% of price)", GroupName = "Filters", Order = 2)]
        public double MaxIBRangePct { get; set; }
    }
}
