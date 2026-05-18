require "net/smtp"
begin
  smtp = Net::SMTP.new(ENV["SMTP_ADDRESS"], ENV["SMTP_PORT"].to_i)
  # smtp.enable_starttls # Explicitly disabled for port 25 test
  smtp.start(ENV["SMTP_DOMAIN"], ENV["SMTP_USERNAME"], ENV["SMTP_PASSWORD"], ENV["SMTP_AUTHENTICATION"].to_sym)
  puts "SMTP_OK"
  smtp.finish
rescue => e
  puts "SMTP_ERROR: #{e.message}"
end
