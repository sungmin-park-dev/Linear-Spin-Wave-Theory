-- pandoc Lua filter used for the 2026-10-11 conversion of the LSWT research notes to Markdown.
-- Display math becomes $$ ... $$ with \tag{n}; \eqref/\ref are resolved to those numbers
-- or to the section title; the original \label names are kept as HTML comments.
local nums, count, heads = {}, 0, {}
local function conv(text)
  local env = text:match('^%s*\\begin{([%a]+%*?)}')
  if not env then return text, {} end
  local starred = env:sub(-1) == '*'
  local base = env:gsub('%*','')
  local body = text:match('^%s*\\begin{[%a%*]+}(.*)\\end{[%a%*]+}%s*$') or text
  local labels = {}
  body = body:gsub('\\label{([^}]*)}', function(l) labels[#labels+1]=l; return '' end)
  body = body:gsub('^%s+',''):gsub('%s+$','')
  local tag = ''
  if not starred and (#labels > 0 or base == 'equation') then
    count = count + 1
    for _,l in ipairs(labels) do nums[l] = count end
    tag = ' \\tag{' .. count .. '}'
  end
  if base == 'align' or base == 'gather' or base == 'multline' then
    local inner = (base == 'gather') and 'gathered' or 'aligned'
    return '\\begin{' .. inner .. '}' .. body .. '\\end{' .. inner .. '}' .. tag, labels
  end
  return body .. tag, labels
end
function Pandoc(doc)
  doc = doc:walk{ Header = function(h) heads[h.identifier] = pandoc.utils.stringify(h.content) end }
  doc = doc:walk{ Math = function(m)
    if m.mathtype ~= 'DisplayMath' then return nil end
    local t, labels = conv(m.text)
    local out = { pandoc.Math('DisplayMath', t) }
    if #labels > 0 then
      table.insert(out, 1, pandoc.RawInline('html', '<!-- eq ' .. count .. ': ' .. table.concat(labels, ', ') .. ' -->'))
    end
    return out
  end }
  doc = doc:walk{ Div = function(d)
    local c = d.classes[1] or ''
    if c == 'minipage' then return d.content end
    if c == 'tcolorbox' or c == 'snugshade' then return pandoc.BlockQuote(d.content) end
  end }
  return doc:walk{ Link = function(l)
    if not l.attributes['reference-type'] then return nil end
    local id = l.attributes['reference']
    if nums[id] then return pandoc.Str('(' .. nums[id] .. ')') end
    if heads[id] then return pandoc.Emph{pandoc.Str(heads[id])} end
    return pandoc.Str('[?' .. id .. ']')
  end }
end
