# Wide Log Examples

Ниже — примеры в формате JSONL (1 строка = 1 event). Их удобно копировать в `wide-log-examples.jsonl`.

## Example 1: HTTP success
{"event_type":"wide_event","schema_version":"1","timestamp":"2026-02-10T10:30:00Z","level":"info","service":{"name":"orders-api","version":"2.1.0","env":"prod","region":"eu-central-1"},"correlation":{"request_id":"req_8bf7ec2d","trace_id":"abc123","span_id":"def456"},"http":{"method":"GET","path":"/api/orders","status_code":200},"outcome":"success","duration_ms":234,"actor":{"user_id":"usr_123","org_id":"org_9","tier":"premium"},"db":{"query_count":2,"total_ms":80,"slowest_ms":45,"error_count":0},"cache":{"hit_count":1,"miss_count":0},"ext":{"call_count":0,"total_ms":0}}

## Example 2: HTTP error (payment)
{"event_type":"wide_event","schema_version":"1","timestamp":"2026-02-10T10:31:12Z","level":"error","service":{"name":"checkout-service","version":"2.4.1","env":"prod","region":"eu-central-1"},"correlation":{"request_id":"req_91aa12","trace_id":"t9c1","span_id":"s77"},"http":{"method":"POST","path":"/api/checkout","status_code":500},"outcome":"error","duration_ms":1247,"actor":{"user_id":"usr_456","org_id":"org_12","tier":"premium"},"feature_flags":{"new_checkout":true},"payment":{"provider":"stripe","attempt":3},"db":{"query_count":3,"total_ms":930,"slowest_ms":847,"error_count":1},"ext":{"call_count":1,"total_ms":220,"targets":["stripe"]},"error":{"type":"PaymentFailed","code":"CARD_DECLINED","message":"card_declined","retriable":false}}

## Example 3: gRPC slow path
{"event_type":"wide_event","schema_version":"1","timestamp":"2026-02-10T10:33:05Z","level":"info","service":{"name":"order-service","version":"5.0.0","env":"prod","region":"eu-central-1"},"correlation":{"request_id":"req_77","trace_id":"tr_55","span_id":"sp_99"},"rpc":{"system":"grpc","service":"OrderService","method":"CreateOrder","status_code":"OK"},"outcome":"success","duration_ms":2780,"actor":{"user_id":"usr_888","org_id":"org_2","tier":"enterprise"},"db":{"query_count":7,"total_ms":2100,"slowest_ms":1200,"error_count":0},"ext":{"call_count":2,"total_ms":500,"targets":["inventory-service","pricing-service"]}}

## Example 4: background job
{"event_type":"wide_event","schema_version":"1","timestamp":"2026-02-10T11:00:00Z","level":"info","service":{"name":"mailer-worker","version":"1.9.3","env":"prod","region":"eu-central-1"},"correlation":{"request_id":"job_44"},"job":{"job_name":"weekly_email_digest","job_id":"job_44","attempt":1,"queue":"emails"},"outcome":"success","duration_ms":90500,"tenant":{"tenant_id":"t_12"},"counters":{"emails_sent":12034,"emails_failed":12},"ext":{"call_count":1,"total_ms":3400,"targets":["email-provider"]}}

## Query recipes (псевдо-SQL/лог-поиск)
1) Все ошибки за час:
   WHERE event_type='wide_event' AND outcome='error' AND timestamp > now()-1h

2) Топ самых медленных:
   WHERE event_type='wide_event' ORDER BY duration_ms DESC LIMIT 100

3) Ошибки по tier:
   WHERE outcome='error' GROUP BY actor.tier COUNT(*)

4) Фильтр по endpoint:
   WHERE http.path='/api/checkout' AND outcome='error'

5) По конкретному request_id/trace_id:
   WHERE correlation.request_id='req_91aa12' OR correlation.trace_id='t9c1'

6) Деградация внешней зависимости:
   WHERE ext.targets CONTAINS 'stripe' ORDER BY ext.total_ms DESC
