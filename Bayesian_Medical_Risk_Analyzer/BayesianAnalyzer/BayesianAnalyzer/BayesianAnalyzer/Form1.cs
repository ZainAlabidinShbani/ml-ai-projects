// =============================================================================
//  BayesianAnalyzer.cs  –  Bayesian Medical Risk Analyzer
//  Single-file WinForms application targeting .NET 4.0 / Visual Studio 2010.
//
//  Architecture overview:
//    AppConstants            – application-wide magic numbers and labels
//    TrainingSample          – one labeled feature vector
//    ModelSettings           – hyperparameters (prior, smoothing, threshold …)
//    GaussianClassModel      – per-class statistics after training
//    BayesianPrediction      – result of a single Predict() call
//    EvaluationReport        – aggregated confusion-matrix metrics
//    TrainTestReport         – both train and test EvaluationReports together
//    TrainedBayesianClassifier – Naive-Bayes classifier (log-space arithmetic)
//    ModelTrainer            – fits a classifier from labeled samples
//    ModelEvaluator          – LOO and train/test evaluation routines
//    DiagnosticSession       – thin façade: train → predict → evaluate → reset
//    RiskBar                 – custom progress-bar control for probability display
//    Form1                   – main WinForms UI
// =============================================================================

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace BayesianAnalyzer
{
    // =========================================================================
    //  ENTRY POINT
    // =========================================================================

    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }

    // =========================================================================
    //  CONSTANTS
    // =========================================================================

    internal static class AppConstants
    {
        public const string PositiveLabel = "Patient";
        public const string NegativeLabel = "Healthy";
        public const int MinTotalSamples = 6;
        public const int MinSamplesPerClass = 3;
        public const double OutlierZScore = 3.0;
        public const double MinEpsilon = 1e-6;
        public const double MinPrior = 1e-12;
        public const double MinWeight = 1e-12;
    }

    // =========================================================================
    //  CUSTOM CONTROL  –  RiskBar
    // =========================================================================

    /// <summary>
    /// Horizontal filled-bar control that displays a risk percentage.
    /// The fill color and width update via SetValue().
    /// </summary>
    public class RiskBar : Panel
    {
        private readonly Panel _fill;
        private readonly Label _label;
        private double _pct;

        public RiskBar()
        {
            BackColor = Color.FromArgb(226, 232, 240);
            BorderStyle = BorderStyle.FixedSingle;
            DoubleBuffered = true;

            _fill = new Panel
            {
                Dock = DockStyle.Left,
                Width = 0,
                BackColor = Color.FromArgb(22, 163, 74)
            };

            _label = new Label
            {
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 11F, FontStyle.Bold)
            };

            Controls.Add(_label);
            Controls.Add(_fill);
            Resize += delegate { RefreshBar(); };
            SetValue(0, Color.FromArgb(22, 163, 74));
        }

        /// <summary>Updates the displayed percentage and bar color.</summary>
        public void SetValue(double percent, Color color)
        {
            _pct = Math.Max(0, Math.Min(100, percent));
            _fill.BackColor = color;
            _label.Text = _pct.ToString("F2", CultureInfo.InvariantCulture) + "%";
            RefreshBar();
        }

        private void RefreshBar()
        {
            _fill.Width = (int)Math.Round(Width * (_pct / 100.0));
            _fill.SendToBack();
        }
    }

    // =========================================================================
    //  DATA MODELS
    // =========================================================================

    /// <summary>A single labeled example with a fixed-length feature vector.</summary>
    public class TrainingSample
    {
        public double[] Features { get; set; }
        public string Label { get; set; }

        public TrainingSample(double[] features, string label)
        {
            Features = features;
            Label = label;
        }
    }

    /// <summary>Hyperparameters that control training and inference behaviour.</summary>
    public class ModelSettings
    {
        public bool UseEmpiricalPrior { get; set; }
        public double ManualPositivePrior { get; set; }
        public double VarianceSmoothing { get; set; }
        public double DecisionThreshold { get; set; }
        public double PositiveClassWeight { get; set; }
        public string PositiveLabel { get; set; }
        public string NegativeLabel { get; set; }
    }

    /// <summary>Gaussian distribution parameters learned for one class.</summary>
    public class GaussianClassModel
    {
        public string Label { get; set; }
        public int SampleCount { get; set; }
        public double[] Mean { get; set; }
        public double[] Variance { get; set; }
        public double[] StdDev { get; set; }
        public double Prior { get; set; }
    }

    /// <summary>Result returned by a single Predict() call.</summary>
    public class BayesianPrediction
    {
        public string PredictedLabel { get; set; }
        public Dictionary<string, double> Posterior { get; set; }
        public double PositivePosterior { get; set; }
        public bool IsOutlier { get; set; }
    }

    /// <summary>Aggregated classification metrics derived from a confusion matrix.</summary>
    public class EvaluationReport
    {
        public int TP { get; set; }
        public int TN { get; set; }
        public int FP { get; set; }
        public int FN { get; set; }
        public int EvaluatedSamples { get; set; }
        public int SkippedFolds { get; set; }
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1 { get; set; }
    }

    /// <summary>Combines train-set and test-set evaluation results with sample counts.</summary>
    public class TrainTestReport
    {
        public EvaluationReport TrainReport { get; set; }
        public EvaluationReport TestReport { get; set; }
        public int TrainCount { get; set; }
        public int TestCount { get; set; }
    }

    // =========================================================================
    //  CORE ML  –  Classifier
    // =========================================================================

    /// <summary>
    /// Gaussian Naive Bayes classifier.
    /// Prediction uses log-space arithmetic (log-sum-exp) to prevent underflow
    /// with many features or very small likelihoods.
    /// </summary>
    public class TrainedBayesianClassifier
    {
        private readonly Dictionary<string, GaussianClassModel> _classModels;
        private readonly ModelSettings _settings;
        private readonly int _featureCount;
        private readonly double _epsilon;

        public TrainedBayesianClassifier(
            Dictionary<string, GaussianClassModel> classModels,
            ModelSettings settings,
            int featureCount,
            double epsilon)
        {
            _classModels = classModels;
            _settings = settings;
            _featureCount = featureCount;
            _epsilon = epsilon;
        }

        // Read-only accessors used by the UI and save/load logic
        public Dictionary<string, GaussianClassModel> ClassModels { get { return _classModels; } }
        public double Epsilon { get { return _epsilon; } }
        public int FeatureCount { get { return _featureCount; } }
        public ModelSettings Settings { get { return _settings; } }

        /// <summary>
        /// Classifies <paramref name="features"/> and returns posterior probabilities
        /// and the winning label.
        /// </summary>
        public BayesianPrediction Predict(double[] features)
        {
            var logScores = new Dictionary<string, double>();

            foreach (var model in _classModels.Values)
            {
                // Start with the class log-prior
                double prior = model.Prior > 0 ? model.Prior : AppConstants.MinPrior;
                double logJoint = Math.Log(prior);

                // Optional asymmetric cost multiplier for the positive class
                if (model.Label == _settings.PositiveLabel)
                {
                    double w = _settings.PositiveClassWeight > 0
                        ? _settings.PositiveClassWeight
                        : AppConstants.MinWeight;
                    logJoint += Math.Log(w);
                }

                // Accumulate Gaussian log-likelihood for each feature dimension
                for (int i = 0; i < _featureCount; i++)
                {
                    double variance = model.Variance[i] > 0 ? model.Variance[i] : _epsilon;
                    double diff = features[i] - model.Mean[i];
                    logJoint += -0.5 * (Math.Log(2.0 * Math.PI * variance) + (diff * diff) / variance);
                }

                logScores[model.Label] = logJoint;
            }

            // Normalise to proper probabilities using the log-sum-exp trick
            double logEvidence = LogSumExp(logScores.Values);
            var posterior = new Dictionary<string, double>();

            foreach (var kv in logScores)
                posterior[kv.Key] = Math.Exp(kv.Value - logEvidence);

            double posPosterior = posterior.ContainsKey(_settings.PositiveLabel)
                ? posterior[_settings.PositiveLabel]
                : 0.0;

            return new BayesianPrediction
            {
                Posterior = posterior,
                PositivePosterior = posPosterior,
                PredictedLabel = posPosterior >= _settings.DecisionThreshold
                    ? _settings.PositiveLabel
                    : _settings.NegativeLabel,
                IsOutlier = CheckIsOutlier(features)
            };
        }

        /// <summary>
        /// Returns true when the mean per-feature absolute z-score (averaged across all class models)
        /// exceeds <see cref="AppConstants.OutlierZScore"/>.
        /// </summary>
        private bool CheckIsOutlier(double[] features)
        {
            double best = double.MaxValue;
            double minStd = Math.Sqrt(_epsilon);

            foreach (var model in _classModels.Values)
            {
                double sum = 0.0;
                for (int i = 0; i < _featureCount; i++)
                {
                    double std = model.StdDev[i] < minStd ? minStd : model.StdDev[i];
                    sum += Math.Abs(features[i] - model.Mean[i]) / std;
                }
                best = Math.Min(best, sum / _featureCount);
            }

            return best > AppConstants.OutlierZScore;
        }

        /// <summary>Numerically stable log-sum-exp: max + log( sum( exp(v - max) ) ).</summary>
        private static double LogSumExp(IEnumerable<double> values)
        {
            double max = values.Max();
            double sum = 0.0;
            foreach (double v in values)
                sum += Math.Exp(v - max);
            return max + Math.Log(sum);
        }
    }

    // =========================================================================
    //  CORE ML  –  Trainer
    // =========================================================================

    /// <summary>
    /// Fits a Gaussian Naive Bayes model from labeled training samples.
    /// Variance smoothing (added to each feature variance) prevents numerical
    /// collapse on small or perfectly-separated datasets.
    /// </summary>
    public class ModelTrainer
    {
        public TrainedBayesianClassifier Train(List<TrainingSample> samples, ModelSettings settings)
        {
            if (samples == null || samples.Count == 0)
                throw new InvalidOperationException("No training samples provided.");

            int featureCount = samples[0].Features.Length;

            // Group samples by class label
            var grouped = samples
                .GroupBy(s => s.Label)
                .ToDictionary(g => g.Key, g => g.ToList());

            if (!grouped.ContainsKey(settings.PositiveLabel) ||
                !grouped.ContainsKey(settings.NegativeLabel))
                throw new InvalidOperationException(
                    "Both classes (" + settings.PositiveLabel + " and " +
                    settings.NegativeLabel + ") must be present in the training set.");

            int totalCount = samples.Count;
            double globalMaxVariance = 0.0;
            var models = new Dictionary<string, GaussianClassModel>();

            // ── First pass: compute per-class means and sample variances ────────
            foreach (var pair in grouped)
            {
                string label = pair.Key;
                List<TrainingSample> classSamples = pair.Value;
                var mean = new double[featureCount];
                var variance = new double[featureCount];
                var stdDev = new double[featureCount];

                for (int i = 0; i < featureCount; i++)
                {
                    var vals = classSamples.Select(s => s.Features[i]).ToList();
                    mean[i] = vals.Average();
                    variance[i] = ComputeSampleVariance(vals);

                    if (variance[i] > globalMaxVariance)
                        globalMaxVariance = variance[i];
                }

                // Set prior: empirical frequency or user-supplied manual value
                double prior = settings.UseEmpiricalPrior
                    ? (double)classSamples.Count / totalCount
                    : (label == settings.PositiveLabel
                        ? settings.ManualPositivePrior
                        : 1.0 - settings.ManualPositivePrior);

                models[label] = new GaussianClassModel
                {
                    Label = label,
                    SampleCount = classSamples.Count,
                    Mean = mean,
                    Variance = variance,
                    StdDev = stdDev,
                    Prior = prior
                };
            }

            // ── Second pass: apply smoothing, compute standard deviations ───────
            double epsilon = Math.Max(
                settings.VarianceSmoothing * (globalMaxVariance > 0 ? globalMaxVariance : 1.0),
                AppConstants.MinEpsilon);

            foreach (var model in models.Values)
            {
                for (int i = 0; i < featureCount; i++)
                {
                    model.Variance[i] += epsilon;
                    model.StdDev[i] = Math.Sqrt(model.Variance[i]);
                }
            }

            return new TrainedBayesianClassifier(models, settings, featureCount, epsilon);
        }

        /// <summary>Bessel-corrected (n − 1) sample variance. Returns 0 for fewer than 2 values.</summary>
        private static double ComputeSampleVariance(List<double> values)
        {
            if (values == null || values.Count <= 1)
                return 0.0;

            double mean = values.Average();
            double sum = 0.0;
            foreach (double v in values)
            {
                double d = v - mean;
                sum += d * d;
            }
            return sum / (values.Count - 1);
        }
    }

    // =========================================================================
    //  CORE ML  –  Evaluator
    // =========================================================================

    /// <summary>
    /// Provides Leave-One-Out (LOO) cross-validation and train/test split evaluation.
    /// Folds that leave fewer than <see cref="AppConstants.MinSamplesPerClass"/> examples
    /// per class in the training fold are silently skipped.
    /// </summary>
    public static class ModelEvaluator
    {
        // ── Leave-One-Out cross-validation ─────────────────────────────────────

        public static EvaluationReport EvaluateLeaveOneOut(
            List<TrainingSample> samples,
            ModelSettings settings)
        {
            var report = new EvaluationReport();
            if (samples == null || samples.Count < 3)
                return report;

            int tp = 0, tn = 0, fp = 0, fn = 0, skipped = 0;
            var trainer = new ModelTrainer();
            var buffer = new List<TrainingSample>(samples.Count - 1);

            for (int i = 0; i < samples.Count; i++)
            {
                // Build fold: all samples except index i
                buffer.Clear();
                for (int j = 0; j < samples.Count; j++)
                    if (j != i) buffer.Add(samples[j]);

                int posCount = buffer.Count(s => s.Label == settings.PositiveLabel);
                int negCount = buffer.Count(s => s.Label == settings.NegativeLabel);

                // Skip this fold if either class is under-represented
                if (posCount < AppConstants.MinSamplesPerClass ||
                    negCount < AppConstants.MinSamplesPerClass)
                {
                    skipped++;
                    continue;
                }

                var model = trainer.Train(buffer, settings);
                var pred = model.Predict(samples[i].Features);

                bool actualPos = samples[i].Label == settings.PositiveLabel;
                bool predictedPos = pred.PredictedLabel == settings.PositiveLabel;

                if (actualPos && predictedPos) tp++;
                else if (!actualPos && !predictedPos) tn++;
                else if (!actualPos && predictedPos) fp++;
                else fn++;
            }

            int total = tp + tn + fp + fn;
            PopulateReport(report, tp, tn, fp, fn, total, skipped);
            return report;
        }

        // ── Train / test set evaluation ────────────────────────────────────────

        public static TrainTestReport EvaluateTrainTest(
            List<TrainingSample> train,
            List<TrainingSample> test,
            ModelSettings settings)
        {
            var result = new TrainTestReport
            {
                TrainCount = train != null ? train.Count : 0,
                TestCount = test != null ? test.Count : 0,
                TrainReport = new EvaluationReport(),
                TestReport = new EvaluationReport()
            };

            if (train == null || test == null || train.Count == 0 || test.Count == 0)
                return result;

            var model = new ModelTrainer().Train(train, settings);
            result.TrainReport = EvaluateSet(model, train);
            result.TestReport = EvaluateSet(model, test);
            return result;
        }

        private static EvaluationReport EvaluateSet(
            TrainedBayesianClassifier model,
            List<TrainingSample> samples)
        {
            int tp = 0, tn = 0, fp = 0, fn = 0;

            foreach (var s in samples)
            {
                var pred = model.Predict(s.Features);
                bool actualPos = s.Label == model.Settings.PositiveLabel;
                bool predictedPos = pred.PredictedLabel == model.Settings.PositiveLabel;

                if (actualPos && predictedPos) tp++;
                else if (!actualPos && !predictedPos) tn++;
                else if (!actualPos && predictedPos) fp++;
                else fn++;
            }

            int total = tp + tn + fp + fn;
            var report = new EvaluationReport();
            PopulateReport(report, tp, tn, fp, fn, total, 0);
            return report;
        }

        /// <summary>Fills all metric fields of <paramref name="report"/> from raw counts.</summary>
        private static void PopulateReport(
            EvaluationReport report,
            int tp, int tn, int fp, int fn,
            int total, int skipped)
        {
            report.TP = tp;
            report.TN = tn;
            report.FP = fp;
            report.FN = fn;
            report.EvaluatedSamples = total;
            report.SkippedFolds = skipped;
            report.Accuracy = total > 0 ? (double)(tp + tn) / total : 0.0;
            report.Precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
            report.Recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
            report.F1 = (report.Precision + report.Recall) > 0
                ? 2.0 * report.Precision * report.Recall / (report.Precision + report.Recall)
                : 0.0;
        }
    }

    // =========================================================================
    //  SESSION FAÇADE
    // =========================================================================

    /// <summary>
    /// Manages the lifecycle of one trained model: Train → Predict → Evaluate → Reset.
    /// Keeps training logic decoupled from the UI.
    /// </summary>
    public class DiagnosticSession
    {
        private TrainedBayesianClassifier _classifier;

        public bool IsTrained { get { return _classifier != null; } }
        public TrainedBayesianClassifier Classifier { get { return _classifier; } }

        public void Train(List<TrainingSample> samples, ModelSettings settings)
        {
            _classifier = new ModelTrainer().Train(samples, settings);
        }

        public BayesianPrediction Predict(double[] features)
        {
            if (_classifier == null)
                throw new InvalidOperationException("Model has not been trained yet.");
            return _classifier.Predict(features);
        }

        public EvaluationReport EvaluateLOO(List<TrainingSample> samples, ModelSettings settings)
        {
            return ModelEvaluator.EvaluateLeaveOneOut(samples, settings);
        }

        public TrainTestReport EvaluateTrainTest(
            List<TrainingSample> train,
            List<TrainingSample> test,
            ModelSettings settings)
        {
            return ModelEvaluator.EvaluateTrainTest(train, test, settings);
        }

        public void Reset() { _classifier = null; }
    }

    // =========================================================================
    //  MAIN FORM
    // =========================================================================

    public class Form1 : Form
    {
        // ── Control references ─────────────────────────────────────────────────

        // Tab control
        private TabControl _tabs;

        // Training tab
        private DataGridView _trainingGrid;
        private CheckBox _empiricalPriorCheck;
        private TrackBar _priorSlider, _smoothingSlider, _thresholdSlider, _weightSlider;
        private Label _priorValueLabel, _smoothingValueLabel, _thresholdValueLabel, _weightValueLabel;
        private Label _datasetSummaryLabel;
        private Button _addRowButton, _removeRowButton, _loadDemoButton, _clearButton;
        private Button _importButton, _exportButton, _saveModelButton, _loadModelButton;
        private Button _trainButton, _splitButton;

        // Inference tab
        private TrackBar _glucoseSlider, _ageSlider, _bmiSlider, _bpSlider;
        private Label _glucoseValueLabel, _ageValueLabel, _bmiValueLabel, _bpValueLabel;
        private Button _classifyButton;
        private Label _diagnosisLabel, _patientPosteriorLabel, _healthyPosteriorLabel, _outlierLabel;
        private RiskBar _riskBar;
        private Chart _posteriorChart, _featureChart;

        // Analytics tab
        private Label _healthyStatsLabel, _patientStatsLabel, _metricsLabel, _confusionLabel;
        private Chart _pieChart, _confusionChart;

        // Status bar
        private Label _modelStatusLabel, _statusBarLabel;
        private ToolTip _toolTip;

        // ── Application state ──────────────────────────────────────────────────

        private readonly DiagnosticSession _session = new DiagnosticSession();
        private List<TrainingSample> _trainSet = new List<TrainingSample>();
        private List<TrainingSample> _testSet = new List<TrainingSample>();
        private TrainTestReport _lastTTReport;

        // ── Color palette ──────────────────────────────────────────────────────

        private readonly Color _colorApp = Color.FromArgb(248, 250, 252);
        private readonly Color _colorCard = Color.White;
        private readonly Color _colorInset = Color.FromArgb(241, 245, 249);
        private readonly Color _colorAccent = Color.FromArgb(37, 99, 235);
        private readonly Color _colorSuccess = Color.FromArgb(22, 163, 74);
        private readonly Color _colorDanger = Color.FromArgb(220, 38, 38);
        private readonly Color _colorWarning = Color.FromArgb(245, 158, 11);
        private readonly Color _colorText = Color.FromArgb(15, 23, 42);
        private readonly Color _colorMuted = Color.FromArgb(100, 116, 139);
        private readonly Color _colorHeader = Color.FromArgb(228, 238, 255);

        // =========================================================================
        //  CONSTRUCTOR
        // =========================================================================

        public Form1()
        {
            Text = "Bayesian Medical Risk Analyzer";
            StartPosition = FormStartPosition.CenterScreen;
            Size = new Size(1500, 900);
            MinimumSize = new Size(1200, 750);
            Font = new Font("Segoe UI", 9F);
            BackColor = _colorApp;

            _toolTip = new ToolTip
            {
                AutoPopDelay = 5000,
                InitialDelay = 400,
                ReshowDelay = 200
            };

            BuildInterface();
            LoadDemoData();
            UpdateSliderLabels();
            SetModelStatus("Model not trained  —  load data, then click Train/Test Split");
        }

        // =========================================================================
        //  TOP-LEVEL LAYOUT
        // =========================================================================

        private void BuildInterface()
        {
            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(8),
                ColumnCount = 1,
                RowCount = 3
            };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 68F));   // header
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));   // tabs
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 28F));   // status bar

            root.Controls.Add(BuildHeaderPanel(), 0, 0);
            root.Controls.Add(BuildTabsPanel(), 0, 1);
            root.Controls.Add(BuildStatusBar(), 0, 2);

            Controls.Add(root);
        }

        // ─── Header ───────────────────────────────────────────────────────────

        private Control BuildHeaderPanel()
        {
            var panel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = _colorCard,
                BorderStyle = BorderStyle.FixedSingle
            };

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 2,
                Padding = new Padding(12, 4, 12, 4)
            };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 65F));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 35F));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 55F));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 45F));

            var title = new Label
            {
                Text = "Bayesian Medical Risk Analyzer",
                Font = new Font("Segoe UI", 15F, FontStyle.Bold),
                ForeColor = _colorText,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.BottomLeft
            };

            var sub = new Label
            {
                Text = "Gaussian Naïve Bayes  ·  Train/Test Split  ·  LOO Cross-Validation  ·  CSV Import/Export  ·  Model Save/Load",
                Font = new Font("Segoe UI", 8.5F),
                ForeColor = _colorMuted,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.TopLeft
            };

            _modelStatusLabel = new Label
            {
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                ForeColor = _colorAccent,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleRight
            };

            layout.Controls.Add(title, 0, 0);
            layout.Controls.Add(sub, 0, 1);
            layout.SetRowSpan(_modelStatusLabel, 2);
            layout.Controls.Add(_modelStatusLabel, 1, 0);

            panel.Controls.Add(layout);
            return panel;
        }

        // ─── Tab control ───────────────────────────────────────────────────────

        private Control BuildTabsPanel()
        {
            _tabs = new TabControl
            {
                Dock = DockStyle.Fill,
                Padding = new Point(14, 5),
                Font = new Font("Segoe UI", 9F)
            };

            _tabs.TabPages.Add(MakeTab("  Training  ", BuildTrainingTab()));
            _tabs.TabPages.Add(MakeTab("  Inference  ", BuildInferenceTab()));
            _tabs.TabPages.Add(MakeTab("  Analytics  ", BuildAnalyticsTab()));
            _tabs.TabPages.Add(MakeTab("  Model Lab  ", BuildModelLabTab()));

            return _tabs;
        }

        private static TabPage MakeTab(string name, Control content)
        {
            var page = new TabPage(name) { BackColor = Color.FromArgb(248, 250, 252) };
            content.Dock = DockStyle.Fill;
            page.Controls.Add(content);
            return page;
        }

        // ─── Status bar ────────────────────────────────────────────────────────

        private Control BuildStatusBar()
        {
            var bar = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = _colorHeader,
                Padding = new Padding(8, 0, 8, 0)
            };

            _statusBarLabel = new Label
            {
                Text = "Ready.",
                Dock = DockStyle.Fill,
                ForeColor = _colorText,
                Font = new Font("Segoe UI", 8.5F),
                TextAlign = ContentAlignment.MiddleLeft
            };

            bar.Controls.Add(_statusBarLabel);
            return bar;
        }

        // =========================================================================
        //  TAB: TRAINING
        // =========================================================================

        private Control BuildTrainingTab()
        {
            var card = MakeCard();

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 5
            };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 26F));   // section title
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 55F));   // data grid
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40F));   // buttons
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 22F));   // summary + prior checkbox
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 45F));   // sliders

            root.Controls.Add(MakeSectionTitle("Training Data"), 0, 0);
            root.Controls.Add(BuildTrainingGrid(), 0, 1);
            root.Controls.Add(BuildTrainingButtons(), 0, 2);
            root.Controls.Add(BuildSummaryAndCheckbox(), 0, 3);
            root.Controls.Add(BuildSettingsSliders(), 0, 4);

            card.Controls.Add(root);
            return card;
        }

        private Control BuildTrainingGrid()
        {
            _trainingGrid = new DataGridView
            {
                Dock = DockStyle.Fill,
                AllowUserToAddRows = true,
                AllowUserToDeleteRows = true,
                RowHeadersVisible = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                MultiSelect = false,
                BackgroundColor = Color.White,
                BorderStyle = BorderStyle.FixedSingle,
                EditMode = DataGridViewEditMode.EditOnEnter,
                GridColor = Color.FromArgb(218, 225, 238)
            };

            _trainingGrid.EnableHeadersVisualStyles = false;
            var headerStyle = _trainingGrid.ColumnHeadersDefaultCellStyle;
            headerStyle.BackColor = _colorHeader;
            headerStyle.ForeColor = _colorText;
            headerStyle.Font = new Font("Segoe UI", 9F, FontStyle.Bold);

            var selStyle = _trainingGrid.RowsDefaultCellStyle;
            selStyle.SelectionBackColor = Color.FromArgb(210, 225, 255);
            selStyle.SelectionForeColor = _colorText;

            _trainingGrid.Columns.Add("Glucose", "Glucose (mg/dL)");
            _trainingGrid.Columns.Add("Age", "Age (yr)");
            _trainingGrid.Columns.Add("BMI", "BMI");
            _trainingGrid.Columns.Add("BloodPressure", "Blood Pressure (mmHg)");

            var classCol = new DataGridViewComboBoxColumn
            {
                Name = "Class",
                HeaderText = "Class"
            };
            classCol.Items.Add(AppConstants.NegativeLabel);
            classCol.Items.Add(AppConstants.PositiveLabel);
            _trainingGrid.Columns.Add(classCol);

            // Color rows by class: green tint for Healthy, red tint for Patient
            _trainingGrid.RowPostPaint += OnGridRowPostPaint;
            _trainingGrid.CellValueChanged += delegate { RefreshSummaryLabel(); };

            return _trainingGrid;
        }

        private void OnGridRowPostPaint(object sender, DataGridViewRowPostPaintEventArgs e)
        {
            if (e.RowIndex < 0 || e.RowIndex >= _trainingGrid.Rows.Count) return;
            var row = _trainingGrid.Rows[e.RowIndex];
            if (row.IsNewRow) return;

            string label = CellText(row, 4);
            if (label == AppConstants.NegativeLabel) row.DefaultCellStyle.BackColor = Color.FromArgb(240, 253, 244);
            else if (label == AppConstants.PositiveLabel) row.DefaultCellStyle.BackColor = Color.FromArgb(254, 242, 242);
            else row.DefaultCellStyle.BackColor = Color.White;
        }

        private Control BuildTrainingButtons()
        {
            var flow = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                WrapContents = false,
                Padding = new Padding(0, 4, 0, 4),
                AutoScroll = true
            };

            _addRowButton = MakeButton("+ Add Row", _colorAccent, 90);
            _addRowButton.Click += delegate
            {
                _trainingGrid.Rows.Add("100", "35", "24", "120", AppConstants.NegativeLabel);
                RefreshSummaryLabel();
            };

            _removeRowButton = MakeButton("− Remove", _colorWarning, 88);
            _removeRowButton.Click += delegate
            {
                if (_trainingGrid.SelectedRows.Count > 0 && !_trainingGrid.SelectedRows[0].IsNewRow)
                {
                    _trainingGrid.Rows.Remove(_trainingGrid.SelectedRows[0]);
                    RefreshSummaryLabel();
                }
            };

            _loadDemoButton = MakeButton("Load Demo", _colorSuccess, 100);
            _loadDemoButton.Click += delegate { LoadDemoData(); };

            _clearButton = MakeButton("Clear All", Color.FromArgb(108, 112, 124), 88);
            _clearButton.Click += delegate
            {
                _trainingGrid.Rows.Clear();
                _trainSet.Clear();
                _testSet.Clear();
                _lastTTReport = null;
                _session.Reset();
                SetModelStatus("Model reset — no data");
                RefreshSummaryLabel();
                RefreshAllVisuals();
            };

            _importButton = MakeButton("Import CSV", _colorAccent, 100);
            _importButton.Click += delegate { ImportCsv(); };

            _exportButton = MakeButton("Export CSV", _colorAccent, 100);
            _exportButton.Click += delegate { ExportCsv(); };

            _saveModelButton = MakeButton("Save Model", _colorSuccess, 104);
            _saveModelButton.Click += delegate { SaveModel(); };

            _loadModelButton = MakeButton("Load Model", _colorSuccess, 104);
            _loadModelButton.Click += delegate { LoadModel(); };

            // Separator panel between data and training buttons
            var sep = new Panel { Width = 12, Height = 1, BackColor = Color.Transparent };

            _trainButton = MakeButton("Train / Test Split", _colorAccent, 150);
            _trainButton.Click += delegate { TrainWithSplit(); };

            _splitButton = MakeButton("LOO Eval", _colorWarning, 100);
            _splitButton.Click += delegate { TrainWithLOO(); };

            // Register tooltips
            _toolTip.SetToolTip(_trainButton, "Stratified 80/20 split: train on 80%, evaluate on held-out 20%");
            _toolTip.SetToolTip(_splitButton, "Leave-One-Out cross-validation across all samples");
            _toolTip.SetToolTip(_importButton, "Import CSV with columns: Glucose, Age, BMI, BloodPressure, Status");
            _toolTip.SetToolTip(_exportButton, "Export the current grid to a CSV file");
            _toolTip.SetToolTip(_saveModelButton, "Save the trained Bayesian model to a .mdl file");
            _toolTip.SetToolTip(_loadModelButton, "Restore a previously saved .mdl model file");

            flow.Controls.AddRange(new Control[]
            {
                _addRowButton, _removeRowButton, _loadDemoButton, _clearButton,
                sep,
                _importButton, _exportButton, _saveModelButton, _loadModelButton,
                new Panel { Width = 12 },
                _trainButton, _splitButton
            });

            return flow;
        }

        private Control BuildSummaryAndCheckbox()
        {
            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2
            };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));

            _datasetSummaryLabel = new Label
            {
                Text = "No data loaded.",
                Dock = DockStyle.Fill,
                ForeColor = _colorMuted,
                TextAlign = ContentAlignment.MiddleLeft
            };

            _empiricalPriorCheck = new CheckBox
            {
                Text = "Use empirical prior (derived from class frequencies)",
                Checked = true,
                AutoSize = false,
                Dock = DockStyle.Fill,
                ForeColor = _colorText,
                TextAlign = ContentAlignment.MiddleLeft
            };
            _empiricalPriorCheck.CheckedChanged += delegate
            {
                _priorSlider.Enabled = !_empiricalPriorCheck.Checked;
                UpdateSliderLabels();
            };
            _toolTip.SetToolTip(_empiricalPriorCheck,
                "When checked the prior equals the class-frequency ratio in the training set.\n" +
                "Uncheck to control the positive-class prior manually via the slider.");

            layout.Controls.Add(_datasetSummaryLabel, 0, 0);
            layout.Controls.Add(_empiricalPriorCheck, 1, 0);
            return layout;
        }

        private Control BuildSettingsSliders()
        {
            var grid = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 2
            };
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            grid.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));
            grid.RowStyles.Add(new RowStyle(SizeType.Percent, 50F));

            var c1 = MakeSliderCard(
                "Manual Patient Prior (%)",
                out _priorSlider, out _priorValueLabel,
                1, 99, 50,
                "Positive-class prior probability used when empirical prior is disabled.");

            var c2 = MakeSliderCard(
                "Variance Smoothing",
                out _smoothingSlider, out _smoothingValueLabel,
                0, 100, 25,
                "Laplace-style additive constant on variance. Prevents zero-variance collapse.");

            var c3 = MakeSliderCard(
                "Decision Threshold (%)",
                out _thresholdSlider, out _thresholdValueLabel,
                10, 90, 50,
                "Posterior probability cutoff above which a sample is labelled Patient.");

            var c4 = MakeSliderCard(
                "Patient Class Weight",
                out _weightSlider, out _weightValueLabel,
                50, 200, 100,
                "Asymmetric cost multiplier on the positive-class log-prior (sensitivity vs. specificity trade-off).");

            c1.Dock = c2.Dock = c3.Dock = c4.Dock = DockStyle.Fill;

            // Wire live-update for all setting sliders
            foreach (var tb in new[] { _priorSlider, _smoothingSlider, _thresholdSlider, _weightSlider })
                tb.Scroll += delegate { UpdateSliderLabels(); };

            // Respect initial empirical-prior state
            _priorSlider.Enabled = false;

            grid.Controls.Add(c1, 0, 0);
            grid.Controls.Add(c2, 1, 0);
            grid.Controls.Add(c3, 0, 1);
            grid.Controls.Add(c4, 1, 1);
            return grid;
        }

        // =========================================================================
        //  TAB: INFERENCE
        // =========================================================================

        private Control BuildInferenceTab()
        {
            var card = MakeCard();

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 4
            };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 26F));   // title
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 84F));   // feature sliders
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 38F));   // classify button
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));   // result + charts

            root.Controls.Add(MakeSectionTitle("Live Patient Inference"), 0, 0);
            root.Controls.Add(BuildInferenceSliders(), 0, 1);
            root.Controls.Add(BuildClassifyRow(), 0, 2);
            root.Controls.Add(BuildInferenceBody(), 0, 3);

            card.Controls.Add(root);
            return card;
        }

        private Control BuildInferenceSliders()
        {
            var row = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 4,
                RowCount = 1
            };
            for (int i = 0; i < 4; i++)
                row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25F));

            var g = MakeSliderCard("Glucose", out _glucoseSlider, out _glucoseValueLabel, 50, 250, 120, "Fasting plasma glucose (mg/dL)");
            var a = MakeSliderCard("Age", out _ageSlider, out _ageValueLabel, 18, 90, 45, "Patient age in years");
            var b = MakeSliderCard("BMI", out _bmiSlider, out _bmiValueLabel, 15, 50, 25, "Body-mass index (kg/m²)");
            var p = MakeSliderCard("Blood Pressure", out _bpSlider, out _bpValueLabel, 80, 200, 120, "Systolic blood pressure (mmHg)");

            g.Dock = a.Dock = b.Dock = p.Dock = DockStyle.Fill;

            _glucoseSlider.Scroll += OnInferenceSliderChanged;
            _ageSlider.Scroll += OnInferenceSliderChanged;
            _bmiSlider.Scroll += OnInferenceSliderChanged;
            _bpSlider.Scroll += OnInferenceSliderChanged;

            row.Controls.Add(g, 0, 0);
            row.Controls.Add(a, 1, 0);
            row.Controls.Add(b, 2, 0);
            row.Controls.Add(p, 3, 0);
            return row;
        }

        private Control BuildClassifyRow()
        {
            var flow = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                WrapContents = false,
                Padding = new Padding(0, 4, 0, 4)
            };

            _classifyButton = MakeButton("Classify Patient", _colorAccent, 160);
            _toolTip.SetToolTip(_classifyButton,
                "Runs Bayesian inference on the current slider values.\n" +
                "If no model is trained, a Train/Test Split is performed first.");
            _classifyButton.Click += OnClassifyClicked;

            var hint = new Label
            {
                Text = "Move any slider to auto-classify when a model is trained.",
                ForeColor = _colorMuted,
                AutoSize = true,
                Margin = new Padding(10, 8, 0, 0)
            };

            flow.Controls.Add(_classifyButton);
            flow.Controls.Add(hint);
            return flow;
        }

        private Control BuildInferenceBody()
        {
            var body = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2
            };
            body.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 58F));
            body.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 42F));

            // ── Left: classification result panel ────────────────────────────
            var result = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = _colorInset,
                BorderStyle = BorderStyle.FixedSingle,
                Padding = new Padding(10)
            };

            _diagnosisLabel = new Label
            {
                Text = "NOT CLASSIFIED",
                BackColor = _colorAccent,
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 22F, FontStyle.Bold),
                TextAlign = ContentAlignment.MiddleCenter,
                Dock = DockStyle.Top,
                Height = 62
            };

            _riskBar = new RiskBar
            {
                Dock = DockStyle.Top,
                Height = 34,
                Margin = new Padding(0, 6, 0, 4)
            };

            _patientPosteriorLabel = new Label
            {
                Text = "Patient posterior:  —",
                Dock = DockStyle.Top,
                Height = 26,
                Font = new Font("Segoe UI", 11F, FontStyle.Bold),
                ForeColor = _colorDanger
            };

            _healthyPosteriorLabel = new Label
            {
                Text = "Healthy posterior:  —",
                Dock = DockStyle.Top,
                Height = 26,
                Font = new Font("Segoe UI", 11F, FontStyle.Bold),
                ForeColor = _colorSuccess
            };

            _outlierLabel = new Label
            {
                Text = "Train the model first.",
                Dock = DockStyle.Top,
                Height = 22,
                ForeColor = _colorMuted
            };

            // Add in reverse order because DockStyle.Top stacks bottom-up
            result.Controls.Add(_outlierLabel);
            result.Controls.Add(_healthyPosteriorLabel);
            result.Controls.Add(_patientPosteriorLabel);
            result.Controls.Add(_riskBar);
            result.Controls.Add(_diagnosisLabel);

            // ── Right: charts ───────────────────────────────────────────────
            var charts = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 2
            };
            charts.RowStyles.Add(new RowStyle(SizeType.Percent, 55F));
            charts.RowStyles.Add(new RowStyle(SizeType.Percent, 45F));

            _posteriorChart = BuildLineChart(
                "P(Patient) vs Glucose", "Glucose (mg/dL)", "P(Patient)", _colorAccent);
            _featureChart = BuildColumnChart(
                "Current Feature Values", "Feature", "Value");

            _posteriorChart.Dock = DockStyle.Fill;
            _featureChart.Dock = DockStyle.Fill;

            charts.Controls.Add(_posteriorChart, 0, 0);
            charts.Controls.Add(_featureChart, 0, 1);

            body.Controls.Add(result, 0, 0);
            body.Controls.Add(charts, 1, 0);
            return body;
        }

        // =========================================================================
        //  TAB: ANALYTICS
        // =========================================================================

        private Control BuildAnalyticsTab()
        {
            var card = MakeCard();

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 3
            };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 26F));   // title
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 40F));   // stats cards
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 60F));   // charts

            root.Controls.Add(MakeSectionTitle("Analytics, Validation & Confusion Matrix"), 0, 0);
            root.Controls.Add(BuildAnalyticsStatsRow(), 0, 1);
            root.Controls.Add(BuildAnalyticsChartsRow(), 0, 2);

            card.Controls.Add(root);
            return card;
        }

        private Control BuildAnalyticsStatsRow()
        {
            var row = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 3,
                RowCount = 1,
                Padding = new Padding(0, 4, 0, 4)
            };
            row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.3F));
            row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.3F));
            row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.3F));

            var healthyCard = MakeLabelCard("Healthy Class Statistics", out _healthyStatsLabel,
                "Not trained yet.");
            var patientCard = MakeLabelCard("Patient Class Statistics", out _patientStatsLabel,
                "Not trained yet.");
            var metricsCard = MakeLabelCard("Evaluation Metrics", out _metricsLabel,
                "Train the model to view validation metrics.");

            healthyCard.Dock = patientCard.Dock = metricsCard.Dock = DockStyle.Fill;

            row.Controls.Add(healthyCard, 0, 0);
            row.Controls.Add(patientCard, 1, 0);
            row.Controls.Add(metricsCard, 2, 0);
            return row;
        }

        private Control BuildAnalyticsChartsRow()
        {
            var row = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2
            };
            row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 42F));
            row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 58F));

            _pieChart = BuildPieChart();
            _pieChart.Dock = DockStyle.Fill;

            // Confusion chart wrapped with a detail label below it
            _confusionChart = BuildConfusionChart();
            _confusionChart.Dock = DockStyle.Fill;

            _confusionLabel = new Label
            {
                Dock = DockStyle.Bottom,
                Height = 52,
                ForeColor = _colorMuted,
                Text = "Confusion matrix values will appear here after training.",
                Padding = new Padding(6, 4, 0, 0),
                Font = new Font("Segoe UI", 9F)
            };

            var confPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.White,
                BorderStyle = BorderStyle.FixedSingle
            };
            confPanel.Controls.Add(_confusionChart);
            confPanel.Controls.Add(_confusionLabel);

            row.Controls.Add(_pieChart, 0, 0);
            row.Controls.Add(confPanel, 1, 0);
            return row;
        }

        // =========================================================================
        //  TAB: MODEL LAB
        // =========================================================================

        private Control BuildModelLabTab()
        {
            var card = MakeCard();

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 2
            };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 26F));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));

            root.Controls.Add(MakeSectionTitle("Model Lab  —  Quick Actions & Workflow Guide"), 0, 0);
            root.Controls.Add(BuildModelLabBody(), 0, 1);

            card.Controls.Add(root);
            return card;
        }

        private Control BuildModelLabBody()
        {
            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2
            };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 32F));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 68F));

            // ── Left: quick actions ─────────────────────────────────────────
            var leftBox = new GroupBox
            {
                Text = "Quick Actions",
                Dock = DockStyle.Fill,
                BackColor = Color.White,
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                Padding = new Padding(10)
            };

            var leftFlow = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoScroll = true,
                Padding = new Padding(2)
            };

            Button btnDemo = MakeButton("Load Demo & Train", _colorSuccess, 180);
            Button btnReset = MakeButton("Reset Model", _colorDanger, 180);
            Button btnInfer = MakeButton("Go to Inference", _colorAccent, 180);
            Button btnAnal = MakeButton("Go to Analytics", _colorAccent, 180);

            foreach (var b in new[] { btnDemo, btnReset, btnInfer, btnAnal })
                b.Margin = new Padding(0, 0, 0, 6);

            btnDemo.Click += delegate { LoadDemoData(); TrainWithSplit(); };
            btnReset.Click += delegate
            {
                _session.Reset();
                _lastTTReport = null;
                _diagnosisLabel.Text = "NOT CLASSIFIED";
                _diagnosisLabel.BackColor = _colorAccent;
                SetModelStatus("Model reset");
                RefreshAllVisuals();
            };
            btnInfer.Click += delegate { _tabs.SelectedIndex = 1; };
            btnAnal.Click += delegate { _tabs.SelectedIndex = 2; };

            leftFlow.Controls.AddRange(new Control[] { btnDemo, btnReset, btnInfer, btnAnal });

            leftBox.Controls.Add(leftFlow);

            // ── Right: workflow guide ───────────────────────────────────────
            var rightBox = new GroupBox
            {
                Text = "Recommended Workflow",
                Dock = DockStyle.Fill,
                BackColor = Color.White,
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                Padding = new Padding(12)
            };

            var steps = new string[]
            {
                "1.  Prepare data:  Import your CSV (columns: Glucose, Age, BMI, BloodPressure, Status) or click Load Demo.",
                "2.  Tune hyperparameters:  Use the sliders on the Training tab (smoothing, threshold, class weight).",
                "3.  Train the model:  Click 'Train / Test Split' for a quick evaluation, or 'LOO Eval' for thorough cross-validation.",
                "4.  Review metrics:  Switch to the Analytics tab — check class statistics, the confusion matrix, and F1 score.",
                "5.  Run inference:  Go to the Inference tab and move the patient sliders — results update in real-time.",
                "6.  Save the model:  Click 'Save Model' to store the trained parameters in a .mdl file for later reuse.",
                "7.  Export data:    Click 'Export CSV' to save the current dataset for version control or sharing."
            };

            var stepsPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoScroll = true
            };

            foreach (string step in steps)
            {
                stepsPanel.Controls.Add(new Label
                {
                    Text = step,
                    AutoSize = false,
                    Dock = DockStyle.None,
                    Size = new Size(700, 28),
                    ForeColor = _colorText,
                    Font = new Font("Segoe UI", 9.5F)
                });
            }

            rightBox.Controls.Add(stepsPanel);

            layout.Controls.Add(leftBox, 0, 0);
            layout.Controls.Add(rightBox, 1, 0);
            return layout;
        }

        // =========================================================================
        //  REUSABLE UI HELPERS
        // =========================================================================

        /// <summary>White card panel with a thin border, used as each tab's root container.</summary>
        private Panel MakeCard()
        {
            return new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = _colorCard,
                BorderStyle = BorderStyle.FixedSingle,
                Padding = new Padding(10)
            };
        }

        private Label MakeSectionTitle(string text)
        {
            return new Label
            {
                Text = text,
                Font = new Font("Segoe UI", 11F, FontStyle.Bold),
                ForeColor = _colorText,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft
            };
        }

        private Button MakeButton(string text, Color backColor, int width)
        {
            var b = new Button
            {
                Text = text,
                Width = width,
                Height = 28,
                FlatStyle = FlatStyle.Flat,
                BackColor = backColor,
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 8.5F, FontStyle.Bold),
                Cursor = Cursors.Hand
            };
            b.FlatAppearance.BorderSize = 0;
            return b;
        }

        /// <summary>
        /// Creates a compact slider card: title (left) and live value (right) on the top row,
        /// with the TrackBar filling the remaining space — zero overlap between labels.
        /// </summary>
        private Panel MakeSliderCard(
            string title,
            out TrackBar slider,
            out Label valueLabel,
            int min, int max, int value,
            string tooltip = "")
        {
            var panel = new Panel
            {
                BackColor = _colorInset,
                BorderStyle = BorderStyle.FixedSingle,
                Padding = new Padding(6, 4, 6, 2)
            };

            // Top row: title on left, live value on right
            var header = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                Height = 20,
                ColumnCount = 2,
                RowCount = 1,
                BackColor = Color.Transparent
            };
            header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 60F));
            header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 40F));

            var titleLabel = new Label
            {
                Text = title,
                Font = new Font("Segoe UI", 8.5F, FontStyle.Bold),
                ForeColor = _colorText,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft
            };

            valueLabel = new Label
            {
                Font = new Font("Segoe UI", 8F),
                ForeColor = _colorMuted,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleRight
            };

            header.Controls.Add(titleLabel, 0, 0);
            header.Controls.Add(valueLabel, 1, 0);

            // Slider below the header
            slider = new TrackBar
            {
                Minimum = min,
                Maximum = max,
                Value = value,
                TickStyle = TickStyle.None,
                Dock = DockStyle.Fill,
                BackColor = _colorInset,
                SmallChange = 1,
                LargeChange = 5
            };

            if (!string.IsNullOrEmpty(tooltip))
                _toolTip.SetToolTip(slider, tooltip);

            panel.Controls.Add(slider);    // Fill (behind header)
            panel.Controls.Add(header);    // Top (above slider)
            return panel;
        }

        /// <summary>Inset card panel with a bold title label and a content label below it.</summary>
        private Panel MakeLabelCard(string title, out Label contentLabel, string defaultText)
        {
            var panel = new Panel
            {
                BackColor = _colorInset,
                BorderStyle = BorderStyle.FixedSingle,
                Padding = new Padding(8),
                Margin = new Padding(4)
            };

            var titleLbl = new Label
            {
                Text = title,
                Font = new Font("Segoe UI", 9.5F, FontStyle.Bold),
                ForeColor = _colorText,
                Dock = DockStyle.Top,
                Height = 22
            };

            contentLabel = new Label
            {
                Text = defaultText,
                Dock = DockStyle.Fill,
                ForeColor = _colorText,
                Font = new Font("Segoe UI", 9F)
            };

            panel.Controls.Add(contentLabel);
            panel.Controls.Add(titleLbl);
            return panel;
        }

        // ── Chart factories ────────────────────────────────────────────────────

        private Chart BuildLineChart(string title, string xAxis, string yAxis, Color color)
        {
            var c = new Chart { BackColor = Color.White };
            var area = new ChartArea();
            area.AxisX.Title = xAxis;
            area.AxisY.Title = yAxis;
            area.AxisX.MajorGrid.LineColor = Color.FromArgb(220, 225, 235);
            area.AxisY.MajorGrid.LineColor = Color.FromArgb(220, 225, 235);
            c.ChartAreas.Add(area);
            c.Titles.Add(new Title(title)
            {
                Font = new Font("Segoe UI", 8.5F, FontStyle.Bold),
                ForeColor = _colorText
            });
            c.Series.Add(new Series(title)
            {
                ChartType = SeriesChartType.Line,
                Color = color,
                BorderWidth = 2
            });
            return c;
        }

        private Chart BuildColumnChart(string title, string xAxis, string yAxis)
        {
            var c = new Chart { BackColor = Color.White };
            var area = new ChartArea();
            area.AxisX.Title = xAxis;
            area.AxisY.Title = yAxis;
            area.AxisX.MajorGrid.Enabled = false;
            area.AxisY.MajorGrid.LineColor = Color.FromArgb(220, 225, 235);
            c.ChartAreas.Add(area);
            c.Titles.Add(new Title(title)
            {
                Font = new Font("Segoe UI", 8.5F, FontStyle.Bold),
                ForeColor = _colorText
            });
            return c;
        }

        private Chart BuildPieChart()
        {
            var c = new Chart { BackColor = Color.Transparent };
            c.ChartAreas.Add(new ChartArea());
            c.Legends.Add(new Legend { Font = new Font("Segoe UI", 8.5F) });
            c.Titles.Add(new Title("Class Distribution")
            {
                Font = new Font("Segoe UI", 8.5F, FontStyle.Bold),
                ForeColor = _colorText
            });

            var s = new Series("Classes") { ChartType = SeriesChartType.Doughnut };
            s["DoughnutRadius"] = "55";
            s.Points.AddXY(AppConstants.NegativeLabel, 50);
            s.Points.AddXY(AppConstants.PositiveLabel, 50);
            s.Points[0].Color = _colorSuccess;
            s.Points[1].Color = _colorDanger;
            c.Series.Add(s);
            return c;
        }

        private Chart BuildConfusionChart()
        {
            var c = new Chart { BackColor = Color.White };
            var area = new ChartArea();
            area.AxisX.MajorGrid.Enabled = false;
            area.AxisY.MajorGrid.Enabled = false;
            c.ChartAreas.Add(area);
            c.Titles.Add(new Title("Confusion Matrix (Test Set)")
            {
                Font = new Font("Segoe UI", 8.5F, FontStyle.Bold),
                ForeColor = _colorText
            });

            var s = new Series("CM") { ChartType = SeriesChartType.Bar };
            s.Points.AddXY("TP", 0); s.Points[0].Color = _colorSuccess;
            s.Points.AddXY("TN", 0); s.Points[1].Color = Color.FromArgb(134, 197, 134);
            s.Points.AddXY("FP", 0); s.Points[2].Color = _colorDanger;
            s.Points.AddXY("FN", 0); s.Points[3].Color = _colorWarning;
            c.Series.Add(s);
            return c;
        }

        // =========================================================================
        //  DEMO DATA
        // =========================================================================

        /// <summary>
        /// Loads 12 synthetic but medically plausible samples (6 per class)
        /// with clearly separated feature distributions.
        /// </summary>
        private void LoadDemoData()
        {
            _trainingGrid.Rows.Clear();

            // Healthy class — lower glucose, younger, lower BMI and blood pressure
            AddGridRow(72, 24, 21.0, 110, AppConstants.NegativeLabel);
            AddGridRow(78, 26, 22.0, 112, AppConstants.NegativeLabel);
            AddGridRow(84, 29, 23.0, 118, AppConstants.NegativeLabel);
            AddGridRow(90, 31, 24.0, 122, AppConstants.NegativeLabel);
            AddGridRow(98, 33, 25.0, 126, AppConstants.NegativeLabel);
            AddGridRow(106, 36, 26.0, 130, AppConstants.NegativeLabel);

            // Patient class — elevated glucose, older, higher BMI and blood pressure
            AddGridRow(118, 41, 28.0, 138, AppConstants.PositiveLabel);
            AddGridRow(126, 44, 30.0, 146, AppConstants.PositiveLabel);
            AddGridRow(138, 47, 32.0, 154, AppConstants.PositiveLabel);
            AddGridRow(150, 50, 35.0, 162, AppConstants.PositiveLabel);
            AddGridRow(164, 54, 38.0, 170, AppConstants.PositiveLabel);
            AddGridRow(180, 58, 41.0, 178, AppConstants.PositiveLabel);

            RefreshSummaryLabel();
            RefreshAllVisuals();
            UpdateStatusBar("Demo dataset loaded (12 samples, 6 per class).");
        }

        private void AddGridRow(double glucose, double age, double bmi, double bp, string label)
        {
            _trainingGrid.Rows.Add(
                glucose.ToString(CultureInfo.InvariantCulture),
                age.ToString(CultureInfo.InvariantCulture),
                bmi.ToString(CultureInfo.InvariantCulture),
                bp.ToString(CultureInfo.InvariantCulture),
                label);
        }

        // =========================================================================
        //  SLIDER HELPERS
        // =========================================================================

        private void UpdateSliderLabels()
        {
            bool auto = _empiricalPriorCheck != null && _empiricalPriorCheck.Checked;

            if (_priorValueLabel != null) _priorValueLabel.Text = auto ? "Auto" : (_priorSlider.Value + "%");
            if (_smoothingValueLabel != null) _smoothingValueLabel.Text = GetSmoothingValue().ToString("F4", CultureInfo.InvariantCulture);
            if (_thresholdValueLabel != null) _thresholdValueLabel.Text = _thresholdSlider.Value + "%";
            if (_weightValueLabel != null) _weightValueLabel.Text = GetWeightValue().ToString("F2", CultureInfo.InvariantCulture) + "x";

            if (_glucoseValueLabel != null) _glucoseValueLabel.Text = _glucoseSlider.Value + " mg/dL";
            if (_ageValueLabel != null) _ageValueLabel.Text = _ageSlider.Value + " yr";
            if (_bmiValueLabel != null) _bmiValueLabel.Text = _bmiSlider.Value.ToString(CultureInfo.InvariantCulture);
            if (_bpValueLabel != null) _bpValueLabel.Text = _bpSlider.Value + " mmHg";
        }

        private double GetSmoothingValue() { return _smoothingSlider.Value / 1000.0; }
        private double GetWeightValue() { return _weightSlider.Value / 100.0; }
        private double GetThresholdValue() { return _thresholdSlider.Value / 100.0; }
        private double GetPriorValue() { return _priorSlider.Value / 100.0; }

        private ModelSettings BuildSettings()
        {
            return new ModelSettings
            {
                UseEmpiricalPrior = _empiricalPriorCheck.Checked,
                ManualPositivePrior = GetPriorValue(),
                VarianceSmoothing = GetSmoothingValue(),
                DecisionThreshold = GetThresholdValue(),
                PositiveClassWeight = GetWeightValue(),
                PositiveLabel = AppConstants.PositiveLabel,
                NegativeLabel = AppConstants.NegativeLabel
            };
        }

        // =========================================================================
        //  EVENT HANDLERS
        // =========================================================================

        private void OnInferenceSliderChanged(object sender, EventArgs e)
        {
            UpdateSliderLabels();
            if (_session.IsTrained)
                RunInference();
        }

        private void OnClassifyClicked(object sender, EventArgs e)
        {
            if (!_session.IsTrained) TrainWithSplit();
            if (_session.IsTrained) RunInference();
        }

        // =========================================================================
        //  TRAINING WORKFLOWS
        // =========================================================================

        /// <summary>
        /// Stratified 80/20 split: preserves class ratios in both sets,
        /// trains the model on the larger portion, evaluates on the held-out set.
        /// </summary>
        private void TrainWithSplit()
        {
            try
            {
                var samples = ReadSamplesFromGrid();
                ValidateSamples(samples);

                _trainSet = new List<TrainingSample>();
                _testSet = new List<TrainingSample>();
                SplitSamples(samples, 0.8, _trainSet, _testSet);

                var settings = BuildSettings();
                _session.Train(_trainSet, settings);
                _lastTTReport = _session.EvaluateTrainTest(_trainSet, _testSet, settings);

                RefreshAnalyticsFromTrainTest(_session.Classifier, _lastTTReport);

                SetModelStatus(string.Format(CultureInfo.InvariantCulture,
                    "Trained (80/20)  |  Train: {0}  Test: {1}  |  epsilon: {2:F5}",
                    _trainSet.Count, _testSet.Count, _session.Classifier.Epsilon));

                UpdateStatusBar(string.Format(CultureInfo.InvariantCulture,
                    "Train/Test Split  |  Train: {0}  Test: {1}  |  Test Accuracy: {2:F1}%   Precision: {3:F1}%   Recall: {4:F1}%   F1: {5:F1}%",
                    _trainSet.Count, _testSet.Count,
                    _lastTTReport.TestReport.Accuracy * 100,
                    _lastTTReport.TestReport.Precision * 100,
                    _lastTTReport.TestReport.Recall * 100,
                    _lastTTReport.TestReport.F1 * 100));

                RefreshAllVisuals();
                RunInference();
            }
            catch (Exception ex)
            {
                _session.Reset();
                SetModelStatus("Training failed");
                MessageBox.Show(ex.Message, "Training Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        /// <summary>
        /// Trains on the complete dataset then runs Leave-One-Out cross-validation.
        /// Useful when data is too scarce for a separate hold-out test set.
        /// </summary>
        private void TrainWithLOO()
        {
            try
            {
                var samples = ReadSamplesFromGrid();
                ValidateSamples(samples);

                var settings = BuildSettings();
                _session.Train(samples, settings);
                var report = _session.EvaluateLOO(samples, settings);

                RefreshAnalyticsPanel(_session.Classifier, report, samples.Count);

                _trainSet = samples;
                _testSet = new List<TrainingSample>();
                _lastTTReport = null;

                SetModelStatus(string.Format(CultureInfo.InvariantCulture,
                    "LOO  |  Samples: {0}  |  epsilon: {1:F5}",
                    samples.Count, _session.Classifier.Epsilon));

                UpdateStatusBar(string.Format(CultureInfo.InvariantCulture,
                    "Leave-One-Out  |  {0} samples  |  Accuracy: {1:F1}%   F1: {2:F1}%   Skipped folds: {3}",
                    samples.Count, report.Accuracy * 100, report.F1 * 100, report.SkippedFolds));

                RefreshAllVisuals();
                RunInference();
            }
            catch (Exception ex)
            {
                _session.Reset();
                SetModelStatus("Training failed");
                MessageBox.Show(ex.Message, "LOO Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        /// <summary>
        /// Stratified split: keeps class proportions similar in both parts.
        /// Ensures at least <see cref="AppConstants.MinSamplesPerClass"/> per class in each split.
        /// </summary>
        private void SplitSamples(
            List<TrainingSample> samples,
            double ratio,
            List<TrainingSample> train,
            List<TrainingSample> test)
        {
            var healthy = samples.Where(s => s.Label == AppConstants.NegativeLabel).ToList();
            var patient = samples.Where(s => s.Label == AppConstants.PositiveLabel).ToList();

            int hTrain = Math.Max(AppConstants.MinSamplesPerClass, (int)Math.Round(healthy.Count * ratio));
            int pTrain = Math.Max(AppConstants.MinSamplesPerClass, (int)Math.Round(patient.Count * ratio));

            // Leave at least MinSamplesPerClass for the test set
            hTrain = Math.Min(hTrain, healthy.Count - AppConstants.MinSamplesPerClass);
            pTrain = Math.Min(pTrain, patient.Count - AppConstants.MinSamplesPerClass);

            if (hTrain < AppConstants.MinSamplesPerClass || pTrain < AppConstants.MinSamplesPerClass)
                throw new InvalidOperationException(
                    "Not enough samples per class for a valid split. Please add more data.");

            train.AddRange(healthy.Take(hTrain));
            train.AddRange(patient.Take(pTrain));
            test.AddRange(healthy.Skip(hTrain));
            test.AddRange(patient.Skip(pTrain));

            if (test.Count == 0)
                throw new InvalidOperationException("Test set is empty after the split. Add more samples.");
        }

        // =========================================================================
        //  INFERENCE
        // =========================================================================

        private void RunInference()
        {
            if (!_session.IsTrained) return;

            double[] features = GetCurrentPatientFeatures();
            var pred = _session.Predict(features);
            double patientPct = pred.PositivePosterior * 100.0;
            double healthyPct = 100.0 - patientPct;
            bool isPatient = pred.PredictedLabel == AppConstants.PositiveLabel;

            _diagnosisLabel.Text = isPatient ? "PATIENT" : "HEALTHY";
            _diagnosisLabel.BackColor = isPatient ? _colorDanger : _colorSuccess;

            _riskBar.SetValue(patientPct, isPatient ? _colorDanger : _colorSuccess);

            _patientPosteriorLabel.Text = string.Format(CultureInfo.InvariantCulture,
                "Patient posterior:   {0:F2}%", patientPct);
            _healthyPosteriorLabel.Text = string.Format(CultureInfo.InvariantCulture,
                "Healthy posterior:   {0:F2}%", healthyPct);

            if (pred.IsOutlier)
            {
                _outlierLabel.Text = "WARNING: Input lies outside the learned distribution (outlier).";
                _outlierLabel.ForeColor = _colorWarning;
            }
            else
            {
                _outlierLabel.Text = string.Format(CultureInfo.InvariantCulture,
                    "Threshold: {0:F0}%  ·  Class weight: {1:F2}x  ·  Priors applied",
                    GetThresholdValue() * 100.0, GetWeightValue());
                _outlierLabel.ForeColor = _colorMuted;
            }

            UpdateFeatureChart(features);
            RefreshPosteriorCurve();
        }

        private double[] GetCurrentPatientFeatures()
        {
            return new double[]
            {
                (double)_glucoseSlider.Value,
                (double)_ageSlider.Value,
                (double)_bmiSlider.Value,
                (double)_bpSlider.Value
            };
        }

        // =========================================================================
        //  CHART REFRESH
        // =========================================================================

        private void RefreshAllVisuals()
        {
            RefreshPosteriorCurve();
            UpdateFeatureChart(GetCurrentPatientFeatures());
            RefreshPieChart();
            RefreshConfusionMatrix();
        }

        /// <summary>
        /// Plots P(Patient | Glucose) for values 50–250 while holding Age, BMI, and BP
        /// at their current slider positions. Falls back to a sigmoid preview before training.
        /// </summary>
        private void RefreshPosteriorCurve()
        {
            if (_posteriorChart == null) return;

            _posteriorChart.Series.Clear();
            var s = new Series("P(Patient)")
            {
                ChartType = SeriesChartType.Line,
                Color = _colorAccent,
                BorderWidth = 2
            };
            _posteriorChart.Series.Add(s);

            if (_session.IsTrained)
            {
                // Sweep glucose; hold other features fixed
                double[] f = GetCurrentPatientFeatures();
                for (int g = 50; g <= 250; g += 4)
                {
                    f[0] = g;
                    s.Points.AddXY(g, _session.Predict(f).PositivePosterior);
                }
                // Restore the actual glucose value
                f[0] = _glucoseSlider.Value;
            }
            else
            {
                // Sigmoid preview — shows expected shape before any model exists
                for (int g = 50; g <= 250; g += 4)
                    s.Points.AddXY(g, 1.0 / (1.0 + Math.Exp(-((g - 150.0) / 18.0))));
            }
        }

        private void UpdateFeatureChart(double[] features)
        {
            if (_featureChart == null) return;

            _featureChart.Series.Clear();

            string[] names = { "Glucose", "Age", "BMI", "BP" };
            Color[] colors = { _colorAccent, _colorSuccess, _colorWarning, _colorDanger };

            for (int i = 0; i < names.Length; i++)
            {
                var s = new Series(names[i])
                {
                    ChartType = SeriesChartType.Column,
                    Color = colors[i]
                };
                s.Points.AddXY(names[i], features[i]);
                _featureChart.Series.Add(s);
            }
        }

        private void RefreshPieChart()
        {
            if (_pieChart == null) return;

            _pieChart.Series.Clear();
            var s = new Series("Classes") { ChartType = SeriesChartType.Doughnut };
            s["DoughnutRadius"] = "55";

            int healthy = 0, patient = 0;
            foreach (DataGridViewRow row in _trainingGrid.Rows)
            {
                if (row.IsNewRow) continue;
                string l = CellText(row, 4);
                if (l == AppConstants.NegativeLabel) healthy++;
                else if (l == AppConstants.PositiveLabel) patient++;
            }

            if (healthy == 0 && patient == 0) { healthy = 1; patient = 1; }

            s.Points.AddXY(AppConstants.NegativeLabel, healthy);
            s.Points.AddXY(AppConstants.PositiveLabel, patient);
            s.Points[0].Color = _colorSuccess;
            s.Points[1].Color = _colorDanger;

            _pieChart.Series.Add(s);
        }

        private void RefreshConfusionMatrix()
        {
            if (_confusionChart == null) return;

            _confusionChart.Series.Clear();
            var s = new Series("CM") { ChartType = SeriesChartType.Bar };

            int tp = 0, tn = 0, fp = 0, fn = 0;
            if (_lastTTReport != null)
            {
                tp = _lastTTReport.TestReport.TP;
                tn = _lastTTReport.TestReport.TN;
                fp = _lastTTReport.TestReport.FP;
                fn = _lastTTReport.TestReport.FN;
            }

            s.Points.AddXY("TP", tp); s.Points[0].Color = _colorSuccess;
            s.Points.AddXY("TN", tn); s.Points[1].Color = Color.FromArgb(134, 197, 134);
            s.Points.AddXY("FP", fp); s.Points[2].Color = _colorDanger;
            s.Points.AddXY("FN", fn); s.Points[3].Color = _colorWarning;

            _confusionChart.Series.Add(s);
        }

        // =========================================================================
        //  ANALYTICS PANEL UPDATE
        // =========================================================================

        private void RefreshAnalyticsPanel(
            TrainedBayesianClassifier model,
            EvaluationReport report,
            int sampleCount)
        {
            var healthy = model.ClassModels[AppConstants.NegativeLabel];
            var patient = model.ClassModels[AppConstants.PositiveLabel];

            _healthyStatsLabel.Text =
                "Count:     " + healthy.SampleCount + "\n" +
                "Prior:     " + (healthy.Prior * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Mean (avg):" + healthy.Mean.Average().ToString("F2", CultureInfo.InvariantCulture) + "\n" +
                "Variance:  " + healthy.Variance.Average().ToString("F4", CultureInfo.InvariantCulture) + "\n" +
                "Std Dev:   " + healthy.StdDev.Average().ToString("F4", CultureInfo.InvariantCulture);

            _patientStatsLabel.Text =
                "Count:     " + patient.SampleCount + "\n" +
                "Prior:     " + (patient.Prior * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Mean (avg):" + patient.Mean.Average().ToString("F2", CultureInfo.InvariantCulture) + "\n" +
                "Variance:  " + patient.Variance.Average().ToString("F4", CultureInfo.InvariantCulture) + "\n" +
                "Std Dev:   " + patient.StdDev.Average().ToString("F4", CultureInfo.InvariantCulture);

            _metricsLabel.Text =
                "Accuracy:  " + (report.Accuracy * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Precision: " + (report.Precision * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Recall:    " + (report.Recall * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "F1 Score:  " + (report.F1 * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Evaluated: " + report.EvaluatedSamples + "  (Skipped: " + report.SkippedFolds + ")";

            if (_confusionLabel != null)
                _confusionLabel.Text = string.Format(
                    "LOO | Samples: {0}  |  TP: {1}  FP: {2}  TN: {3}  FN: {4}",
                    sampleCount, report.TP, report.FP, report.TN, report.FN);

            RefreshPieChart();
            RefreshConfusionMatrix();
        }

        private void RefreshAnalyticsFromTrainTest(
            TrainedBayesianClassifier model,
            TrainTestReport r)
        {
            var healthy = model.ClassModels[AppConstants.NegativeLabel];
            var patient = model.ClassModels[AppConstants.PositiveLabel];

            _healthyStatsLabel.Text =
                "Count:     " + healthy.SampleCount + "\n" +
                "Prior:     " + (healthy.Prior * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Mean (avg):" + healthy.Mean.Average().ToString("F2", CultureInfo.InvariantCulture) + "\n" +
                "Std Dev:   " + healthy.StdDev.Average().ToString("F4", CultureInfo.InvariantCulture);

            _patientStatsLabel.Text =
                "Count:     " + patient.SampleCount + "\n" +
                "Prior:     " + (patient.Prior * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Mean (avg):" + patient.Mean.Average().ToString("F2", CultureInfo.InvariantCulture) + "\n" +
                "Std Dev:   " + patient.StdDev.Average().ToString("F4", CultureInfo.InvariantCulture);

            _metricsLabel.Text =
                "Train Acc: " + (r.TrainReport.Accuracy * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Test Acc:  " + (r.TestReport.Accuracy * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Precision: " + (r.TestReport.Precision * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "Recall:    " + (r.TestReport.Recall * 100).ToString("F1", CultureInfo.InvariantCulture) + "%\n" +
                "F1 Score:  " + (r.TestReport.F1 * 100).ToString("F1", CultureInfo.InvariantCulture) + "%";

            if (_confusionLabel != null)
                _confusionLabel.Text = string.Format(
                    "80/20 Split | Train: {0}  Test: {1}  |  TP: {2}  FP: {3}  TN: {4}  FN: {5}",
                    r.TrainCount, r.TestCount,
                    r.TestReport.TP, r.TestReport.FP,
                    r.TestReport.TN, r.TestReport.FN);

            _lastTTReport = r;
            RefreshPieChart();
            RefreshConfusionMatrix();
        }

        // =========================================================================
        //  CSV IMPORT / EXPORT
        // =========================================================================

        private void ImportCsv()
        {
            var ofd = new OpenFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
            };
            if (ofd.ShowDialog() != DialogResult.OK) return;

            try
            {
                string[] lines = File.ReadAllLines(ofd.FileName);
                _trainingGrid.Rows.Clear();

                foreach (string line in lines.Skip(1))   // skip header row
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] parts = line.Split(',');
                    if (parts.Length >= 5)
                    {
                        _trainingGrid.Rows.Add(
                            parts[0].Trim(), parts[1].Trim(),
                            parts[2].Trim(), parts[3].Trim(),
                            NormalizeLabel(parts[4].Trim()));
                    }
                }

                RefreshSummaryLabel();
                RefreshAllVisuals();
                UpdateStatusBar("CSV imported: " + ofd.FileName);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Import failed:\n" + ex.Message, "Import Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private void ExportCsv()
        {
            var sfd = new SaveFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv",
                FileName = "dataset.csv"
            };
            if (sfd.ShowDialog() != DialogResult.OK) return;

            try
            {
                var lines = new List<string> { "Glucose,Age,BMI,BloodPressure,Status" };

                foreach (DataGridViewRow row in _trainingGrid.Rows)
                {
                    if (row.IsNewRow) continue;
                    lines.Add(
                        CellText(row, 0) + "," +
                        CellText(row, 1) + "," +
                        CellText(row, 2) + "," +
                        CellText(row, 3) + "," +
                        CellText(row, 4));
                }

                File.WriteAllLines(sfd.FileName, lines.ToArray());
                UpdateStatusBar("Dataset exported: " + sfd.FileName);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Export failed:\n" + ex.Message, "Export Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        // =========================================================================
        //  MODEL SAVE / LOAD
        // =========================================================================

        private void SaveModel()
        {
            if (!_session.IsTrained)
            {
                MessageBox.Show("Train the model first before saving.",
                    "No Model", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            var sfd = new SaveFileDialog
            {
                Filter = "Model files (*.mdl)|*.mdl",
                FileName = "bayes_model.mdl"
            };
            if (sfd.ShowDialog() != DialogResult.OK) return;

            try
            {
                using (var sw = new StreamWriter(sfd.FileName))
                {
                    var clf = _session.Classifier;

                    // File format header + global parameters
                    sw.WriteLine("BayesianAnalyzerModelV2");
                    sw.WriteLine(clf.Epsilon.ToString(CultureInfo.InvariantCulture));
                    sw.WriteLine(clf.FeatureCount.ToString(CultureInfo.InvariantCulture));
                    sw.WriteLine(clf.Settings.UseEmpiricalPrior ? "1" : "0");
                    sw.WriteLine(clf.Settings.ManualPositivePrior.ToString(CultureInfo.InvariantCulture));
                    sw.WriteLine(clf.Settings.VarianceSmoothing.ToString(CultureInfo.InvariantCulture));
                    sw.WriteLine(clf.Settings.DecisionThreshold.ToString(CultureInfo.InvariantCulture));
                    sw.WriteLine(clf.Settings.PositiveClassWeight.ToString(CultureInfo.InvariantCulture));
                    sw.WriteLine(clf.Settings.PositiveLabel);
                    sw.WriteLine(clf.Settings.NegativeLabel);

                    // Per-class model blocks
                    foreach (var kv in clf.ClassModels)
                    {
                        var m = kv.Value;
                        sw.WriteLine("CLASS|" + m.Label + "|" + m.SampleCount + "|" +
                                     m.Prior.ToString(CultureInfo.InvariantCulture));
                        sw.WriteLine("MEAN|" + JoinDoubles(m.Mean));
                        sw.WriteLine("VAR|" + JoinDoubles(m.Variance));
                        sw.WriteLine("STD|" + JoinDoubles(m.StdDev));
                    }
                }

                UpdateStatusBar("Model saved: " + sfd.FileName);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Save failed:\n" + ex.Message, "Save Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private void LoadModel()
        {
            var ofd = new OpenFileDialog { Filter = "Model files (*.mdl)|*.mdl" };
            if (ofd.ShowDialog() != DialogResult.OK) return;

            try
            {
                string[] lines = File.ReadAllLines(ofd.FileName);

                if (lines.Length < 10 || !lines[0].StartsWith("BayesianAnalyzerModel"))
                {
                    MessageBox.Show("Invalid or unsupported model file format.",
                        "Load Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                int idx = 1;
                double epsilon = double.Parse(lines[idx++], CultureInfo.InvariantCulture);
                int featureCount = int.Parse(lines[idx++], CultureInfo.InvariantCulture);
                bool useEmpirical = lines[idx++] == "1";
                double manualPrior = double.Parse(lines[idx++], CultureInfo.InvariantCulture);
                double smoothing = double.Parse(lines[idx++], CultureInfo.InvariantCulture);
                double threshold = double.Parse(lines[idx++], CultureInfo.InvariantCulture);
                double weight = double.Parse(lines[idx++], CultureInfo.InvariantCulture);
                string posLabel = lines[idx++];
                string negLabel = lines[idx++];

                var dict = new Dictionary<string, GaussianClassModel>();

                while (idx < lines.Length)
                {
                    if (!lines[idx].StartsWith("CLASS|")) { idx++; continue; }

                    string[] parts = lines[idx++].Split('|');
                    var m = new GaussianClassModel
                    {
                        Label = parts[1],
                        SampleCount = int.Parse(parts[2]),
                        Prior = double.Parse(parts[3], CultureInfo.InvariantCulture)
                    };
                    m.Mean = ParseVector(lines[idx++].Substring(5), featureCount);
                    m.Variance = ParseVector(lines[idx++].Substring(4), featureCount);
                    m.StdDev = ParseVector(lines[idx++].Substring(4), featureCount);
                    dict[m.Label] = m;
                }

                var settings = new ModelSettings
                {
                    UseEmpiricalPrior = useEmpirical,
                    ManualPositivePrior = manualPrior,
                    VarianceSmoothing = smoothing,
                    DecisionThreshold = threshold,
                    PositiveClassWeight = weight,
                    PositiveLabel = posLabel,
                    NegativeLabel = negLabel
                };

                // Inject the reconstructed classifier directly via reflection
                // (avoids exposing a public setter on DiagnosticSession._classifier)
                typeof(DiagnosticSession)
                    .GetField("_classifier", BindingFlags.NonPublic | BindingFlags.Instance)
                    .SetValue(_session, new TrainedBayesianClassifier(dict, settings, featureCount, epsilon));

                SetModelStatus("Model loaded from file");
                UpdateStatusBar("Model loaded: " + ofd.FileName);
                RefreshAllVisuals();
                RunInference();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Load failed:\n" + ex.Message, "Load Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        // =========================================================================
        //  DATA UTILITIES
        // =========================================================================

        /// <summary>Reads all non-empty, non-header rows from the grid as TrainingSample objects.</summary>
        private List<TrainingSample> ReadSamplesFromGrid()
        {
            var samples = new List<TrainingSample>();

            foreach (DataGridViewRow row in _trainingGrid.Rows)
            {
                if (row.IsNewRow) continue;

                string gt = CellText(row, 0), at = CellText(row, 1),
                       bt = CellText(row, 2), pt = CellText(row, 3),
                       lt = CellText(row, 4);

                // Skip entirely blank rows
                if (gt == "" && at == "" && bt == "" && pt == "" && lt == "") continue;

                double g, a, b, p;
                if (!TryParseDouble(gt, out g)) throw new InvalidOperationException("Invalid Glucose value: " + gt);
                if (!TryParseDouble(at, out a)) throw new InvalidOperationException("Invalid Age value: " + at);
                if (!TryParseDouble(bt, out b)) throw new InvalidOperationException("Invalid BMI value: " + bt);
                if (!TryParseDouble(pt, out p)) throw new InvalidOperationException("Invalid Blood Pressure value: " + pt);
                if (lt == "") throw new InvalidOperationException(
  "Row " + (row.Index + 1) + " is missing a class label.");

                samples.Add(new TrainingSample(new[] { g, a, b, p }, NormalizeLabel(lt)));
            }

            return samples;
        }

        private static void ValidateSamples(List<TrainingSample> samples)
        {
            if (samples.Count < AppConstants.MinTotalSamples)
                throw new InvalidOperationException(
                    "At least " + AppConstants.MinTotalSamples + " samples are required to train.");

            int healthy = samples.Count(s => s.Label == AppConstants.NegativeLabel);
            int patient = samples.Count(s => s.Label == AppConstants.PositiveLabel);

            if (healthy < AppConstants.MinSamplesPerClass || patient < AppConstants.MinSamplesPerClass)
                throw new InvalidOperationException(
                    "At least " + AppConstants.MinSamplesPerClass + " samples per class are required.\n" +
                    "Current — Healthy: " + healthy + "  Patient: " + patient);
        }

        private void RefreshSummaryLabel()
        {
            if (_datasetSummaryLabel == null) return;

            int total = 0, healthy = 0, patient = 0;
            foreach (DataGridViewRow row in _trainingGrid.Rows)
            {
                if (row.IsNewRow) continue;
                string l = CellText(row, 4);
                if (l == AppConstants.NegativeLabel) { healthy++; total++; }
                else if (l == AppConstants.PositiveLabel) { patient++; total++; }
                else if (l != "") total++;
            }

            _datasetSummaryLabel.Text = string.Format(
                "Dataset: {0} rows  ·  Healthy: {1}  ·  Patient: {2}",
                total, healthy, patient);
        }

        // ── Static helpers ─────────────────────────────────────────────────────

        private static bool TryParseDouble(string text, out double value)
        {
            return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value)
                || double.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out value);
        }

        /// <summary>Maps "healthy"/"patient" (case-insensitive) to canonical class labels.</summary>
        private static string NormalizeLabel(string label)
        {
            string lower = label.Trim().ToLowerInvariant();
            if (lower == "healthy") return AppConstants.NegativeLabel;
            if (lower == "patient") return AppConstants.PositiveLabel;
            throw new FormatException(
                "Unknown class label '" + label + "'. Expected: Healthy or Patient.");
        }

        private static string CellText(DataGridViewRow row, int index)
        {
            object v = row.Cells[index].Value;
            return v == null ? "" : v.ToString().Trim();
        }

        private static string JoinDoubles(double[] values)
        {
            return string.Join(";",
                values.Select(x => x.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        private static double[] ParseVector(string s, int count)
        {
            string[] parts = s.Split(';');
            double[] v = new double[count];
            for (int i = 0; i < count && i < parts.Length; i++)
                v[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
            return v;
        }

        // ── Status helpers ─────────────────────────────────────────────────────

        private void SetModelStatus(string text) { _modelStatusLabel.Text = text; }
        private void UpdateStatusBar(string text) { _statusBarLabel.Text = text; }
    }
}
