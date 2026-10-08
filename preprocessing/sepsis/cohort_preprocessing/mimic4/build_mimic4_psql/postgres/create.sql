-------------------------------------------
-- Create the tables and MIMIC-IV schema --
-------------------------------------------

----------------------
-- Creating schemas --
----------------------

-- DROP SCHEMA IF EXISTS mimic_core CASCADE;
-- CREATE SCHEMA mimic_core;
DROP SCHEMA IF EXISTS mimic_hosp CASCADE;
CREATE SCHEMA mimic_hosp;
DROP SCHEMA IF EXISTS mimic_icu CASCADE;
CREATE SCHEMA mimic_icu;
DROP SCHEMA IF EXISTS mimic_note CASCADE;
CREATE SCHEMA mimic_note;
-- DROP SCHEMA IF EXISTS mimic_derived CASCADE;
-- CREATE SCHEMA mimic_derived;

---------------------
-- Creating tables --
---------------------

-- hosp schema

DROP TABLE IF EXISTS mimic_hosp.omr;
CREATE TABLE mimic_hosp.omr
(
  subject_id INTEGER NOT NULL,
  chartdate DATE NOT NULL,
  seq_num INTEGER NOT NULL,
  result_name VARCHAR(100) NOT NULL,
  result_value TEXT NOT NULL
);

DROP TABLE IF EXISTS mimic_hosp.provider;
CREATE TABLE mimic_hosp.provider
(
  provider_id VARCHAR(10) NOT NULL
);

DROP TABLE IF EXISTS mimic_hosp.admissions;
CREATE TABLE mimic_hosp.admissions
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  admittime TIMESTAMP NOT NULL,
  dischtime TIMESTAMP,
  deathtime TIMESTAMP,
  admission_type VARCHAR(40) NOT NULL,
  admit_provider_id VARCHAR(10),
  admission_location VARCHAR(60),
  discharge_location VARCHAR(60),
  insurance VARCHAR(255),
  language VARCHAR(10),
  marital_status VARCHAR(30),
  race VARCHAR(80),
  edregtime TIMESTAMP,
  edouttime TIMESTAMP,
  hospital_expire_flag SMALLINT
);

DROP TABLE IF EXISTS mimic_hosp.patients;
CREATE TABLE mimic_hosp.patients
(
  subject_id INTEGER NOT NULL,
  gender VARCHAR(1) NOT NULL,
  anchor_age INTEGER NOT NULL,
  anchor_year INTEGER NOT NULL,
  anchor_year_group VARCHAR(255) NOT NULL,
  dod TIMESTAMP(0)
);

DROP TABLE IF EXISTS mimic_hosp.transfers;
CREATE TABLE mimic_hosp.transfers
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER,
  transfer_id INTEGER NOT NULL,
  eventtype VARCHAR(10),
  careunit VARCHAR(255),
  intime TIMESTAMP(0),
  outtime TIMESTAMP(0)
);

DROP TABLE IF EXISTS mimic_hosp.d_hcpcs;
CREATE TABLE mimic_hosp.d_hcpcs
(
  code CHAR(5) NOT NULL,
  category DOUBLE PRECISION,
  long_description TEXT,
  short_description VARCHAR(180)
);

DROP TABLE IF EXISTS mimic_hosp.diagnoses_icd;
CREATE TABLE mimic_hosp.diagnoses_icd
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  seq_num INTEGER NOT NULL,
  icd_code CHAR(7),
  icd_version INTEGER
);

DROP TABLE IF EXISTS mimic_hosp.d_icd_diagnoses;
CREATE TABLE mimic_hosp.d_icd_diagnoses
(
  icd_code CHAR(7) NOT NULL,
  icd_version INTEGER NOT NULL,
  long_title VARCHAR(255)
);

DROP TABLE IF EXISTS mimic_hosp.d_icd_procedures;
CREATE TABLE mimic_hosp.d_icd_procedures
(
  icd_code CHAR(7) NOT NULL,
  icd_version INTEGER NOT NULL,
  long_title VARCHAR(255)
);

DROP TABLE IF EXISTS mimic_hosp.d_labitems;
CREATE TABLE mimic_hosp.d_labitems
(
  itemid INTEGER,
  label VARCHAR(50),
  fluid VARCHAR(50),
  category VARCHAR(50)
);

DROP TABLE IF EXISTS mimic_hosp.drgcodes;
CREATE TABLE mimic_hosp.drgcodes
(
  subject_id INTEGER,
  hadm_id INTEGER,
  drg_type VARCHAR(4),
  drg_code VARCHAR(10),
  description VARCHAR(195),
  drg_severity SMALLINT,
  drg_mortality SMALLINT
);

DROP TABLE IF EXISTS mimic_hosp.emar_detail;
CREATE TABLE mimic_hosp.emar_detail
(
  subject_id INTEGER NOT NULL,
  emar_id VARCHAR(25) NOT NULL,
  emar_seq INTEGER NOT NULL,
  parent_field_ordinal VARCHAR(10),
  administration_type VARCHAR(50),
  pharmacy_id INTEGER,
  barcode_type VARCHAR(4),
  reason_for_no_barcode TEXT,
  complete_dose_not_given VARCHAR(5),
  dose_due VARCHAR(100),
  dose_due_unit VARCHAR(50),
  dose_given VARCHAR(255),
  dose_given_unit VARCHAR(50),
  will_remainder_of_dose_be_given VARCHAR(5),
  product_amount_given VARCHAR(30),
  product_unit VARCHAR(30),
  product_code VARCHAR(30),
  product_description VARCHAR(255),
  product_description_other VARCHAR(255),
  prior_infusion_rate VARCHAR(40),
  infusion_rate VARCHAR(40),
  infusion_rate_adjustment VARCHAR(50),
  infusion_rate_adjustment_amount VARCHAR(30),
  infusion_rate_unit VARCHAR(30),
  route VARCHAR(10),
  infusion_complete VARCHAR(1),
  completion_interval VARCHAR(50),
  new_iv_bag_hung VARCHAR(1),
  continued_infusion_in_other_location VARCHAR(1),
  restart_interval TEXT,
  side VARCHAR(10),
  site VARCHAR(255),
  non_formulary_visual_verification VARCHAR(1)
);

DROP TABLE IF EXISTS mimic_hosp.emar;
CREATE TABLE mimic_hosp.emar
(
  subject_id INTEGER NOT NULL,
  hadm_id DOUBLE PRECISION,
  emar_id VARCHAR(25) NOT NULL,
  emar_seq INTEGER NOT NULL,
  poe_id VARCHAR(25) NOT NULL,
  pharmacy_id DOUBLE PRECISION,
  enter_provider_id VARCHAR(10),
  charttime TIMESTAMP NOT NULL,
  medication TEXT,
  event_txt VARCHAR(100),
  scheduletime TIMESTAMP,
  storetime TIMESTAMP NOT NULL
);

DROP TABLE IF EXISTS mimic_hosp.hcpcsevents;
CREATE TABLE mimic_hosp.hcpcsevents
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  chartdate DATE,
  hcpcs_cd CHAR(5) NOT NULL,
  seq_num INTEGER NOT NULL,
  short_description VARCHAR(180)
);

DROP TABLE IF EXISTS mimic_hosp.labevents;
CREATE TABLE mimic_hosp.labevents
(
  labevent_id INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER,
  specimen_id INTEGER NOT NULL,
  itemid INTEGER NOT NULL,
  order_provider_id VARCHAR(10),
  charttime TIMESTAMP(0),
  storetime TIMESTAMP(0),
  value VARCHAR(200),
  valuenum DOUBLE PRECISION,
  valueuom VARCHAR(20),
  ref_range_lower DOUBLE PRECISION,
  ref_range_upper DOUBLE PRECISION,
  flag VARCHAR(10),
  priority VARCHAR(7),
  comments TEXT
);

DROP TABLE IF EXISTS mimic_hosp.microbiologyevents;
CREATE TABLE mimic_hosp.microbiologyevents
(
  microevent_id INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER,
  micro_specimen_id INTEGER NOT NULL,
  order_provider_id VARCHAR(10),
  chartdate TIMESTAMP(0) NOT NULL,
  charttime TIMESTAMP(0),
  spec_itemid INTEGER NOT NULL,
  spec_type_desc VARCHAR(100) NOT NULL,
  test_seq INTEGER NOT NULL,
  storedate TIMESTAMP(0),
  storetime TIMESTAMP(0),
  test_itemid INTEGER,
  test_name VARCHAR(100),
  org_itemid INTEGER,
  org_name VARCHAR(100),
  isolate_num SMALLINT,
  quantity VARCHAR(50),
  ab_itemid INTEGER,
  ab_name VARCHAR(30),
  dilution_text VARCHAR(10),
  dilution_comparison VARCHAR(20),
  dilution_value DOUBLE PRECISION,
  interpretation VARCHAR(5),
  comments TEXT
);

DROP TABLE IF EXISTS mimic_hosp.pharmacy;
CREATE TABLE mimic_hosp.pharmacy
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  pharmacy_id INTEGER NOT NULL,
  poe_id VARCHAR(25),
  starttime TIMESTAMP(3),
  stoptime TIMESTAMP(3),
  medication TEXT,
  proc_type VARCHAR(50) NOT NULL,
  status VARCHAR(50),
  entertime TIMESTAMP(3) NOT NULL,
  verifiedtime TIMESTAMP(3),
  route VARCHAR(50),
  frequency VARCHAR(50),
  disp_sched VARCHAR(255),
  infusion_type VARCHAR(15),
  sliding_scale VARCHAR(1),
  lockout_interval VARCHAR(50),
  basal_rate REAL,
  one_hr_max VARCHAR(10),
  doses_per_24_hrs REAL,
  duration REAL,
  duration_interval VARCHAR(50),
  expiration_value INTEGER,
  expiration_unit VARCHAR(50),
  expirationdate TIMESTAMP(3),
  dispensation VARCHAR(50),
  fill_quantity VARCHAR(50)
);

DROP TABLE IF EXISTS mimic_hosp.poe_detail;
CREATE TABLE mimic_hosp.poe_detail
(
  poe_id VARCHAR(25) NOT NULL,
  poe_seq INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  field_name VARCHAR(255) NOT NULL,
  field_value TEXT
);

DROP TABLE IF EXISTS mimic_hosp.poe;
CREATE TABLE mimic_hosp.poe
(
  poe_id VARCHAR(25) NOT NULL,
  poe_seq INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER,
  ordertime TIMESTAMP(0) NOT NULL,
  order_type VARCHAR(25) NOT NULL,
  order_subtype VARCHAR(50),
  transaction_type VARCHAR(15),
  discontinue_of_poe_id VARCHAR(25),
  discontinued_by_poe_id VARCHAR(25),
  order_provider_id VARCHAR(10),
  order_status VARCHAR(15)
);

DROP TABLE IF EXISTS mimic_hosp.prescriptions;
CREATE TABLE mimic_hosp.prescriptions
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  pharmacy_id INTEGER NOT NULL,
  poe_id VARCHAR(25),
  poe_seq INTEGER,
  order_provider_id VARCHAR(10),
  starttime TIMESTAMP(3),
  stoptime TIMESTAMP(3),
  drug_type VARCHAR(20) NOT NULL,
  drug VARCHAR(255) NOT NULL,
  formulary_drug_cd VARCHAR(50),
  gsn VARCHAR(255),
  ndc VARCHAR(25),
  prod_strength VARCHAR(255),
  form_rx VARCHAR(25),
  dose_val_rx VARCHAR(100),
  dose_unit_rx VARCHAR(50),
  form_val_disp VARCHAR(50),
  form_unit_disp VARCHAR(50),
  doses_per_24_hrs REAL,
  route VARCHAR(50)
);

DROP TABLE IF EXISTS mimic_hosp.procedures_icd;
CREATE TABLE mimic_hosp.procedures_icd
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  seq_num INTEGER NOT NULL,
  chartdate DATE NOT NULL,
  icd_code VARCHAR(7),
  icd_version INTEGER
);

DROP TABLE IF EXISTS mimic_hosp.services;
CREATE TABLE mimic_hosp.services
(
  subject_id INTEGER,
  hadm_id INTEGER,
  transfertime TIMESTAMP(0),
  prev_service VARCHAR(20),
  curr_service VARCHAR(20)
);

-- icu schema

DROP TABLE IF EXISTS mimic_icu.chartevents;
CREATE TABLE mimic_icu.chartevents
(
  subject_id INTEGER,
  hadm_id INTEGER,
  stay_id INTEGER,
  caregiver_id INTEGER,
  charttime TIMESTAMP(0),
  storetime TIMESTAMP(0),
  itemid INTEGER,
  value VARCHAR(200),
  valuenum DOUBLE PRECISION,
  valueuom VARCHAR(20),
  warning SMALLINT
);

DROP TABLE IF EXISTS mimic_icu.datetimeevents;
CREATE TABLE mimic_icu.datetimeevents
(
  subject_id INTEGER,
  hadm_id INTEGER,
  stay_id INTEGER,
  caregiver_id INTEGER,
  charttime TIMESTAMP(3),
  storetime TIMESTAMP(3),
  itemid INTEGER,
  value TIMESTAMP(3),
  valueuom VARCHAR(20),
  warning SMALLINT
);

DROP TABLE IF EXISTS mimic_icu.d_items;
CREATE TABLE mimic_icu.d_items
(
  itemid INTEGER,
  label VARCHAR(200),
  abbreviation VARCHAR(100),
  linksto VARCHAR(50),
  category VARCHAR(100),
  unitname VARCHAR(100),
  param_type VARCHAR(30),
  lownormalvalue FLOAT,
  highnormalvalue FLOAT
);

DROP TABLE IF EXISTS mimic_icu.icustays;
CREATE TABLE mimic_icu.icustays
(
  subject_id INTEGER,
  hadm_id INTEGER,
  stay_id INTEGER,
  first_careunit VARCHAR(80),
  last_careunit VARCHAR(80),
  intime TIMESTAMP(0),
  outtime TIMESTAMP(0),
  los DOUBLE PRECISION
);

DROP TABLE IF EXISTS mimic_icu.inputevents;
CREATE TABLE mimic_icu.inputevents
(
  subject_id INTEGER,
  hadm_id INTEGER,
  stay_id INTEGER,
  caregiver_id INTEGER,
  starttime TIMESTAMP(0),
  endtime TIMESTAMP(0),
  storetime TIMESTAMP(0),
  itemid INTEGER,
  amount DOUBLE PRECISION,
  amountuom VARCHAR(30),
  rate DOUBLE PRECISION,
  rateuom VARCHAR(30),
  orderid BIGINT,
  linkorderid BIGINT,
  ordercategoryname VARCHAR(100),
  secondaryordercategoryname VARCHAR(100),
  ordercomponenttypedescription VARCHAR(200),
  ordercategorydescription VARCHAR(50),
  patientweight DOUBLE PRECISION,
  totalamount DOUBLE PRECISION,
  totalamountuom VARCHAR(50),
  isopenbag SMALLINT,
  continueinnextdept SMALLINT,
  statusdescription VARCHAR(30),
  originalamount DOUBLE PRECISION,
  originalrate DOUBLE PRECISION
);

DROP TABLE IF EXISTS mimic_icu.outputevents;
CREATE TABLE mimic_icu.outputevents
(
  subject_id INTEGER,
  hadm_id INTEGER,
  stay_id INTEGER,
  caregiver_id INTEGER,
  charttime TIMESTAMP(3),
  storetime TIMESTAMP(3),
  itemid INTEGER,
  value DOUBLE PRECISION,
  valueuom VARCHAR(20)
);

DROP TABLE IF EXISTS mimic_icu.procedureevents;
CREATE TABLE mimic_icu.procedureevents
(
  subject_id INTEGER,
  hadm_id INTEGER,
  stay_id INTEGER,
  caregiver_id INTEGER,
  starttime TIMESTAMP,
  endtime TIMESTAMP,
  storetime TIMESTAMP,
  itemid INTEGER,
  value DOUBLE PRECISION,
  valueuom VARCHAR(20),
  location VARCHAR(100),
  locationcategory VARCHAR(50),
  orderid INTEGER,
  linkorderid INTEGER,
  ordercategoryname VARCHAR(50),
  ordercategorydescription VARCHAR(30),
  patientweight DOUBLE PRECISION,
  isopenbag SMALLINT,
  continueinnextdept SMALLINT,
  statusdescription VARCHAR(20),
  originalamount DOUBLE PRECISION,
  originalrate DOUBLE PRECISION
);

DROP TABLE IF EXISTS mimic_icu.caregiver;
CREATE TABLE mimic_icu.caregiver
(
  caregiver_id VARCHAR(10) NOT NULL
);

DROP TABLE IF EXISTS mimic_icu.ingredientevents;
CREATE TABLE mimic_icu.ingredientevents
(
  subject_id INTEGER,
  hadm_id INTEGER,
  stay_id INTEGER,
  caregiver_id INTEGER,
  starttime TIMESTAMP(0),
  endtime TIMESTAMP(0),
  storetime TIMESTAMP(0),
  itemid INTEGER,
  amount DOUBLE PRECISION,
  amountuom VARCHAR(20),
  rate DOUBLE PRECISION,
  rateuom VARCHAR(20),
  orderid INTEGER,
  linkorderid INTEGER,
  statusdescription VARCHAR(20),
  originalamount DOUBLE PRECISION,
  originalrate DOUBLE PRECISION
);


-- note schema
DROP TABLE IF EXISTS mimic_note.discharge;
CREATE TABLE mimic_note.discharge
(
  note_id VARCHAR(25) NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  note_type CHAR(2) NOT NULL,
  note_seq INTEGER NOT NULL,
  charttime TIMESTAMP NOT NULL,
  storetime TIMESTAMP,
  text TEXT NOT NULL
);

DROP TABLE IF EXISTS mimic_note.discharge_detail;
CREATE TABLE mimic_note.discharge_detail
(
  note_id VARCHAR(25) NOT NULL,
  subject_id INTEGER NOT NULL,
  field_name VARCHAR(255) NOT NULL,
  field_value TEXT NOT NULL,
  field_ordinal INTEGER NOT NULL
);


DROP TABLE IF EXISTS mimic_note.radiology;
CREATE TABLE mimic_note.radiology
(
  note_id VARCHAR(25) NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER,
  note_type CHAR(2) NOT NULL,
  note_seq INTEGER NOT NULL,
  charttime TIMESTAMP NOT NULL,
  storetime TIMESTAMP,
  text TEXT NOT NULL
);

DROP TABLE IF EXISTS mimic_note.radiology_detail;
CREATE TABLE mimic_note.radiology_detail
(
  note_id VARCHAR(25) NOT NULL,
  subject_id INTEGER NOT NULL,
  field_name VARCHAR(255) NOT NULL,
  field_value TEXT NOT NULL,
  field_ordinal INTEGER NOT NULL
);