require "net/smtp"
begin
  smtp = Net::SMTP.new(ENV["SMTP_ADDRESS"], ENV["SMTP_PORT"].to_i)
  if ENV["SMTP_ENABLE_STARTTLS_AUTO"] == "true"
    smtp.enable_starttls
  end
  smtp.start(ENV["SMTP_DOMAIN"], ENV["SMTP_USERNAME"], ENV["SMTP_PASSWORD"], ENV["SMTP_AUTHENTICATION"].to_sym)
  puts "SMTP_OK"
  smtp.finish
rescue => e
  puts "SMTP_ERROR: #{e.message}"
end
