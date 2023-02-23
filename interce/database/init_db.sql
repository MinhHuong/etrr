/* drop all existing tables, to use only when initializing the database */
drop table if exists motif cascade ;
drop table if exists ce_query cascade ;
drop table if exists feedback cascade ;
drop table if exists feedback_log cascade ;
drop table if exists marker cascade ;
drop table if exists experiment cascade ;
drop table if exists metrics cascade ;

/* store the official and buffered queries, add "ce_" as prefix because "query" seems to be a predefined keyword */
create table ce_query (
    id              serial primary key,
    fname           varchar not null,
    is_answered     boolean not null default false,
    /* if it is a buffered query (is_buffered = True), then id_ref must have a value */
    is_buffered     boolean not null,
    id_ref          integer
);

/* store the human feedback */
create table feedback (
    id              serial primary key,
    id_query        integer references ce_query(id) not null,
    is_solved       boolean not null default false,
    is_empty        boolean not null
);

/* store the feedback in form of clusters and labels */
create table feedback_log (
    label           varchar(10) not null primary key,
    data            float[][]
);

/* not sure if we will use this table or we will store everything in memory */
create table motif (
    id              serial primary key,
    fname           varchar not null,
    data            integer[][]
);

/* store the markers (one marker = one cycle) */
create table marker (
    id              serial primary key,
    fname           varchar not null,
    mstart          integer not null,
    mend            integer not null,
    no_cyc          integer not null,
    input_len       integer not null,
    cyc_label       varchar,
    cyc_data        integer[][] not null,
    extractor       varchar,
    ext_mode        varchar,
    id_query        integer references ce_query(id),
    id_feedback     integer references feedback(id),
    /* this can be used to create a view that returns official markers only */
    is_official     boolean not null
);

/* a table to store the information of seaprate experiments */
create table if not exists experiment (
    id              serial primary key,
    expe_name       varchar not null,
    expe_date       float not null,
    expe_desc       text
);

/* lastly, a table to store the metrics */
create table if not exists metrics (
    id              serial primary key,
    id_expe         integer references experiment(id) not null,
    /* global measurement */
    n_files         integer not null,
    total_time      float not null,  /* time in terms of seconds*/
    mem_usage       float not null,  /* memory usage in bytes */
    n_cycles        integer not null,  /* total number of cycles extracted so far, in both auto and human mode */
    n_auto_cycles   integer not null,   /* number of cycles extracted automatically */
    n_human_cycles  integer not null,   /* number of cycles selected by the human */
    n_auto          integer not null,   /* number of times InterCE works in auto mode*/
    n_human         integer not null,   /* number of times iNterCE works in human mode */
    is_training     bool not null,      /* whether interce is in a training or testing phase */
    /* memory-related */
    n_motifs                integer not null,   /* number of motifs stored in the memory */
    /* query-related */
    n_total_queries         integer not null,
    n_buffered_queries      integer not null,
    n_official_queries      integer not null,
    /* feedback-related */
    n_feedback                  integer not null,   /* total feedback */
    /* time-related */
    time_m_motifs     float not null,     /* memoty: time to find similar motif (accumulated) */
    time_k_select     float not null,     /* time for the memory K to select cycles automatically (accumulated) */
    time_k_update     float not null,     /* time for the memory K to update the clusters (accumulated) */
    time_ensemble     float not null      /* time for all extractors in the ensemble to identify cycles*/
);

create unique index metrics_expe_nfiles_index on metrics(id_expe, n_files);
