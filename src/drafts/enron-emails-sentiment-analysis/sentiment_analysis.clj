
;; Enable stack traces
;; (clojupyter.misc.stacktrace/set-print-stacktraces! true)
(require '[clojupyter.misc.helper :as helper])

(->> '[[clojure-opennlp "0.5.0"]
       [kixi/stats "0.5.0"]
       [io.forward/clojure-mail "1.0.7"]
       [clojure2d "1.1.0"]
       [metasoarous/oz "1.5.0"]
       [clj-time "0.15.0"]
       [net.cgrand/xforms "0.18.2"]]
     (map helper/add-dependencies)
     doall)

(print (str "Done!"))

;; Load VADER as local repository
;; The vader repo binary must be installed in this directory ./maven-repository
(do
    (use '[cemerick.pomegranate :only (add-dependencies)])
    (add-dependencies 
        :coordinates '[[local/vader "2.0.1"]] 
        :repositories {"local/vader" (str (.toURI (java.io.File. "./maven_repository")))}))

;; Build namespace
(ns drafts.sentiment_analysis
    (:import [net.nunoachenriques.vader SentimentAnalysis]
             [net.nunoachenriques.vader.lexicon English]
             [net.nunoachenriques.vader.text TokenizerEnglish]
             [java.io FileInputStream File]
             [javax.mail Session]
             [javax.mail.internet MimeMessage]
             [java.util Properties])
    (:require [kixi.stats.core :as stats]
              [clojure-mail.core :as mail]
              [clojure-mail.message :refer (read-message)]
              [oz.notebook.clojupyter :as oz]
              [clj-time.core :as t]
              [clj-time.coerce :as c]
              [net.cgrand.xforms :as x])
    (:use [clojure.repl :only (doc source)]
          [clojure.pprint :only (pprint print-table)]
          [opennlp.nlp :only (make-sentence-detector)]))

*ns*

(set! *warn-on-reflection* true)

(def language (English.))
(def tokenizer (TokenizerEnglish.))

(def sa (SentimentAnalysis. language tokenizer))

(. sa (getSentimentAnalysis "Yay!! You are the best!"))

;; Avoiding reflection by type hint:
(. ^SentimentAnalysis sa (getSentimentAnalysis "Yay!! You are the best!"))

(def maildir-path "data/enron_mail/maildir")

(def sample-msg 
    (-> "data/enron_mail/maildir/arnold-j/_sent_mail/36."
        (clojure.java.io/as-file)
        (mail/file->message)
        (read-message)))

(pprint sample-msg)

(defn get-files [start-path re]
    (->> start-path
         (clojure.java.io/as-file)
         (file-seq)
         (map #(.getPath ^File %))
         (filter #(re-matches re %))))

#_(def xform-msg-files
    (comp (map mail/file->message)
          (map read-message)))

(defn raw-message->message-data [m]
    {:to    (-> (get m :to) (first) (get :address))
     :from  (-> (get m :from) (first) (get :address))
     :date-sent (get m :date-sent)
     :date-received (get m :date-received)
     :subject (get m :subject)
     :body  (get-in m [:body :body])})

(def xform-msg-files
    (comp (map mail/file->message)
          (map read-message)
          (map raw-message->message-data)))


(def sent-mail-re #"data\/enron_mail\/maildir\/.*\/_sent_mail\/.*")
(def sent-msg-paths (get-files maildir-path sent-mail-re))

#_(defn msg-reduce
    ([] [])
    ([acc] acc)
    ([acc m]
        (conj acc {:to    (-> (get m :to) (first) (get :address))
                   :from  (-> (get m :from) (first) (get :address))
                   :date-sent (get m :date-sent)
                   :date-received (get m :date-received)
                   :subject (get m :subject)
                   :body  (get-in m [:body :body])})))

#_(def msgs (transduce xform-msg-files msg-reduce sent-msg-paths))

#_(def msgs (into [] xform-msg-files sent-msg-paths))

(def msgs (sequence xform-msg-files sent-msg-paths))

(count msgs)

(defn remove-line-breaks [text]
    (clojure.string/replace text #"\n" ""))

(def get-sentences (make-sentence-detector "./models/en-sent.bin"))

#_(defn add-sentiment
    ([] [])
    ([acc] acc)
    ([acc msg]
      (conj acc (conj msg {:avg-sentiment (->> msg
                                     (:body)
                                     (get-sentences)
                                     (map remove-line-breaks)
                                     (map #(. sa (getSentimentAnalysis %)))
                                     (map #(get % "compound"))
                                     (transduce identity stats/mean))}))))

(defn msg->avg-sentiment [msg]
  (->> msg
       (:body)
       (get-sentences)
       (transduce
        (map (fn [sentence]
               (-> sentence
                   remove-line-breaks
                   (#(. ^SentimentAnalysis sa (getSentimentAnalysis %)))
                   (get "compound"))))
        stats/mean)))

#_(def sentiment (transduce identity add-sentiment (filter #(< (count (get % :body)) 4000) msgs)))

(def sentiment 
    (sequence
          (comp 
            (filter #(< (count (get % :body)) 4000))
            (map (fn [msg] (conj msg {:avg-sentiment (msg->avg-sentiment msg)}))))
          msgs))

#_(pprint (->> (take 10 sentiment)
             (map #(select-keys % [:date-sent :avg-sentiment]))))

(->> sentiment
     (take 10)
     (map #(select-keys % [:date-sent :avg-sentiment]))
     print-table)

(defn same-day? [t1 t2]
    (t/equal? (t/floor t1 t/day) (t/floor t2 t/day)))

#_(def xform-get-time-data
    (comp (map #(select-keys % [:date-sent :avg-sentiment]))
          (map #(hash-map :date (-> (c/from-date (:date-sent %))
                                    (t/floor t/day)
                                    (c/to-date))
                          :avg-sentiment (:avg-sentiment %)))))

(defn get-time-data [{:keys [date-sent avg-sentiment]}]
    {:date (-> date-sent
               c/from-date
               (t/floor t/day)
               (c/to-date))
     :avg-sentiment avg-sentiment})

#_(pprint (eduction xform-get-time-data (take 5 sentiment)))

(->> sentiment
     (eduction (comp (take 5)
                     (map get-time-data)))
     print-table)

#_(defn reduce-daily-sentiment
    ([] {})
    ([acc] 
     (reduce #(conj %1 {(first %2) 
                        (transduce identity stats/mean (second %2))}) (sorted-map) acc))
    ([acc x]
     (let [{date :date sentiment :avg-sentiment} x]
            (if (contains? acc date)
             (update acc date conj sentiment)
             (conj acc {date [sentiment]})))))

#_(def average-sentiment-data (transduce xform-get-time-data reduce-daily-sentiment sentiment))

(def average-sentiment-data (into (sorted-map)
                                  (comp (map get-time-data)
                                        (x/by-key :date
                                                  :avg-sentiment
                                                   x/avg))
                                  sentiment))

(count average-sentiment-data)

(defn average [coll]
  (/ (reduce + coll)
      (count coll)))

(defn moving-average [period coll] 
  (lazy-cat (repeat (dec period) nil) 
            (map average (partition period 1  coll))))

#_(def time-series-data
    (->> average-sentiment-data
         (#(vector (map first %) (map second %)))
         (#(vector (first %) (second %) (moving-average 30 (second %))))
         (apply map vector)
         (map #(hash-map :date (str (nth % 0))
                         :avg-sentiment (nth % 1)
                         :moving-avg (nth % 2)))))

#_(def time-series-data
    (->> average-sentiment-data
         ((juxt keys vals))
         ((fn [[dates values]]
              [(map str dates) values (moving-average 30 values)]))
         (apply map vector)
         (map (partial zipmap [:date :avg-sentiment :moving-avg]))))

(def time-series-data
    (->> average-sentiment-data
         (#(vector (keys %)
                   (vals %)
                   (moving-average 30 (vals %))))
         (apply map (fn [date v smoothed-v]
                        {:date (str date)
                         :avg-sentiment v
                         :moving-avg smoothed-v}))))

#_time-series-data
(print-table time-series-data)

;; (def line-plot
;;   {:data {:values time-series-data}
;;    :width 400
;;    :height 400
;;    :encoding {:x {:field "date", :type "temporal"}
;;               :y {:field "moving-avg"}}
;;    :mark {:type "line" :stroke "red"}})

(def layered-line-plot
    {:width 600
     :height 600
     :data {:values time-series-data}
     :layer [{:mark {:type "line", :stroke "lightblue"}
              :encoding {:x {:field "date", :type "temporal"}
                         :y {:field "avg-sentiment"}}},
             {:mark {:type "line", :stroke "green"}
              :encoding {:x {:field "date", :type "temporal"}
                         :y {:field "moving-avg"}}}]})

;; Render the plot
;; (oz/view! line-plot)
(oz/view! layered-line-plot)
